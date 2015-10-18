#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers_more.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
ImageSegUniformSizeDataLayer<Dtype>::ImageSegUniformSizeDataLayer(
    const LayerParameter& param) :
    BasePrefetchingDataLayer<Dtype>(param) {
}

template<typename Dtype>
ImageSegUniformSizeDataLayer<Dtype>::~ImageSegUniformSizeDataLayer() {
  this->JoinPrefetchThread();
}

template<typename Dtype>
void ImageSegUniformSizeDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand()
        % this->layer_param_.data_param().rand_skip();
    LOG(INFO)<< "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_height = this->layer_param_.transform_param().crop_height();
  int crop_width = this->layer_param_.transform_param().crop_width();

  if (crop_size > 0) {
    CHECK_EQ(crop_height, 0)<< "crop_size and crop_height can not both be non-zero";
    CHECK_EQ(crop_width, 0)
    << "crop_size and crop_width can not both be non-zero";
    crop_height = crop_size;
    crop_width = crop_size;
  }

  int batch_size = this->layer_param_.data_param().batch_size();
  if (crop_height > 0 && crop_width > 0) {
    CHECK_GT(crop_height, 0);
    CHECK_GT(crop_width, 0);
    LOG(INFO)<<"Reshape top blobs according to cropping height and width: "
    <<crop_height<<" x "<<crop_width;

    top[0]->Reshape(batch_size, datum.channels(), crop_height, crop_width);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_height,
        crop_width);
    this->transformed_data_.Reshape(1, datum.channels(), crop_height,
        crop_width);
    if (this->output_labels_) {
      top[1]->Reshape(batch_size, 1, crop_height, crop_width);
      this->prefetch_label_.Reshape(batch_size, 1, crop_height, crop_width);
      this->transformed_label_.Reshape(1, 1, crop_height, crop_width);
    }
  } else {
    int min_height = this->layer_param_.transform_param().min_height();
    int min_width = this->layer_param_.transform_param().min_width();
    CHECK_GT(min_height, 0);
    CHECK_GT(min_width, 0);

    int height_multiple =
        this->layer_param_.transform_param().height_multiple();
    int width_multiple = this->layer_param_.transform_param().width_multiple();
    CHECK_GT(height_multiple, 0);
    CHECK_GT(width_multiple, 0);

    min_height = (min_height / height_multiple) * height_multiple;
    min_width = (min_width / width_multiple) * width_multiple;

    CHECK_GT(min_height, 0);
    CHECK_GT(min_width, 0);

    LOG(INFO)<<"Reshape top blobs according to min height and width: "
    <<min_height<<" x "<<min_width;
    top[0]->Reshape(batch_size, datum.channels(), min_height, min_width);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), min_height,
        min_width);
    this->transformed_data_.Reshape(1, datum.channels(), min_height, min_width);
    if (this->output_labels_) {
      top[1]->Reshape(batch_size, 1, min_height, min_width);
      this->prefetch_label_.Reshape(batch_size, 1, min_height, min_width);
      this->transformed_label_.Reshape(1, 1, min_height, min_width);
    }
  }
  if (top.size() == 3) {
    vector<int> size_blob_shape(1);
    size_blob_shape[0] = 2;
    top[2]->Reshape(size_blob_shape);
    Dtype *size_blob_data = top[2]->mutable_cpu_data();
    size_blob_data[0] = this->prefetch_data_.shape(2);
    size_blob_data[1] = this->prefetch_data_.shape(3);
  }
}

// This function is used to create a thread that prefetches the data
template<typename Dtype>
void ImageSegUniformSizeDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }

  int crop_height = this->layer_param_.transform_param().crop_height();
  int crop_width = this->layer_param_.transform_param().crop_width();

  const int batch_size = this->layer_param_.data_param().batch_size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a blob
    Datum datum;
    datum.ParseFromString(cursor_->value());

    cv::Mat cv_img;
    if (datum.encoded()) {
      cv_img = DecodeDatumToCVMatNative(datum);
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING)<< "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    // Use the aspect ratio in the 1st sample in the mini-batch
    // to decide cropping height and width
    // Make sure training images samples are sorted by aspect ratio
    if (crop_height == 0 && crop_width == 0 && item_id == 0) {
      Dtype aspect_ratio = 0;
      if (datum.encoded()) {
        aspect_ratio = (Dtype) cv_img.rows / (Dtype) cv_img.cols;
      } else {
        aspect_ratio = (Dtype) (datum.height()) / (Dtype) (datum.width());
      }
      this->data_transformer_->ComputeCropHeightWidth(aspect_ratio);

      this->prefetch_data_.Reshape(batch_size, datum.channels(),
          this->data_transformer_->crop_height_from_aspect_ratio(),
          this->data_transformer_->crop_width_from_aspect_ratio());
      this->transformed_data_.Reshape(1, datum.channels(),
          this->data_transformer_->crop_height_from_aspect_ratio(),
          this->data_transformer_->crop_width_from_aspect_ratio());
      if (this->output_labels_) {
        this->prefetch_label_.Reshape(batch_size, 1,
            this->data_transformer_->crop_height_from_aspect_ratio(),
            this->data_transformer_->crop_width_from_aspect_ratio());
        this->transformed_label_.Reshape(1, 1,
            this->data_transformer_->crop_height_from_aspect_ratio(),
            this->data_transformer_->crop_width_from_aspect_ratio());
      }
    }

    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    int offset_data = this->prefetch_data_.offset(item_id);
    int offset_label = this->prefetch_label_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    if (this->output_labels_) {
      this->transformed_label_.set_cpu_data(top_label + offset_label);
    }
    if (datum.encoded()) {
      if (this->output_labels_) {
        this->data_transformer_->TransformImgAndSegUniformSize(datum, cv_img,
            &(this->transformed_data_), &(this->transformed_label_));
      } else {
        this->data_transformer_->TransformImgAndSegUniformSize(datum, cv_img,
            &(this->transformed_data_), NULL);
      }

    } else {
      if (this->output_labels_) {
        this->data_transformer_->TransformImgAndSegUniformSize(datum,
            &(this->transformed_data_), &(this->transformed_label_));
      } else {
        this->data_transformer_->TransformImgAndSegUniformSize(datum,
            &(this->transformed_data_), NULL);
      }
    }
    trans_time += timer.MicroSeconds();
    // go to the next iter
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO)<< "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }  // for (int item_id = 0; item_id < batch_size; ++item_id)
  batch_timer.Stop();
  LOG_FIRST_N(INFO, 10) << "Prefetch batch: " << batch_timer.MilliSeconds()
      << " ms.";
  LOG_FIRST_N(INFO, 10) << "     Read time: " << read_time / 1000 << " ms.";
  LOG_FIRST_N(INFO, 10)
  << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegUniformSizeDataLayer);
REGISTER_LAYER_CLASS(ImageSegUniformSizeData);

}  // namespace caffe
