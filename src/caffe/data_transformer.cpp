#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase) :
    param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0)<<
    "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
    "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }

  min_height_ = param_.min_height();
  min_width_ = param_.min_width();

  CHECK((param_.has_resize_size() &&
      !param_.has_resize_short_side_max() &&
      !param_.has_resize_short_side_min()) ||
      (param_.has_resize_short_side_max() &&
          param_.has_resize_short_side_min() &&
          !param_.has_resize_size()) ||
          (!param_.has_resize_size() &&
              !param_.has_resize_short_side_max() &&
              !param_.has_resize_short_side_min()));
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
    Dtype* transformed_data) {
  const string *data = &datum.data();
  int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();
  int datum_label = datum.label();

  const int resize_short_side_min = param_.resize_short_side_min();
  const int resize_short_side_max = param_.resize_short_side_max();
  const int resize_size = param_.resize_size();
  Datum resized_datum;
  if ((resize_short_side_min > 0 && resize_short_side_max > 0)
      || (resize_size > 0)) {
    int resize_width = 0, resize_height = 0;
    cv::Mat* cv_origin_img = DatumToCVMat(datum);
    if (resize_short_side_min > 0 && resize_short_side_max > 0) {
      // TO DO
      // Now supports only byte data.
      // need to support float data also
      CHECK_GE(resize_short_side_max, resize_short_side_min);
      const int resize_short_side = resize_short_side_min
          + Rand(resize_short_side_max - resize_short_side_min + 1);
      if (cv_origin_img->rows > cv_origin_img->cols) {
        resize_width = resize_short_side;
        resize_height = ceil(
            (float(cv_origin_img->rows) / float(cv_origin_img->cols))
                * resize_width);
      } else {
        resize_height = resize_short_side;
        resize_width = ceil(
            (float(cv_origin_img->cols) / float(cv_origin_img->rows))
                * resize_height);
      }
    } else {
      resize_height = resize_size;
      resize_width = resize_size;
    }
    cv::Mat cv_img;
    cv::resize(*cv_origin_img, cv_img, cv::Size(resize_width, resize_height));
    CVMatToDatum(cv_img, &resized_datum);
    resized_datum.set_label(datum_label);

    datum_channels = resized_datum.channels();
    datum_height = resized_datum.height();
    datum_width = resized_datum.width();
    data = &resized_datum.data();
    delete cv_origin_img;
  }

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = (*data).size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
                                                                                << "Specify either 1 mean_value or as many as channels: "
                                                                                << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
              static_cast<Dtype>(static_cast<uint8_t>((*data)[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] = (datum_element - mean[data_index])
              * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] = (datum_element - mean_values_[c])
                * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
    Blob<Dtype>* transformed_blob) {
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
    Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0)<< "There is no datum to add";
  CHECK_LE(datum_num, num)<<
  "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
    Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0)<< "There is no MAT to add";
  CHECK_EQ(mat_num, num)<<
  "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
    Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  const int resize_short_side_min = param_.resize_short_side_min();
  const int resize_short_side_max = param_.resize_short_side_max();
  const int resize_size = param_.resize_size();
  const cv::Mat *cv_img_ptr = &cv_img;
  cv::Mat cv_resized_img;
  if((resize_short_side_min > 0 && resize_short_side_max > 0) || (resize_size > 0)){
    int resize_width = 0, resize_height = 0;
    if(resize_short_side_min > 0 && resize_short_side_max > 0){
      CHECK_GE(resize_short_side_max, resize_short_side_min);
      const int resize_short_side = resize_short_side_min
          + Rand(resize_short_side_max - resize_short_side_min + 1);
      if(cv_img.rows > cv_img.cols){
        resize_width = resize_short_side;
        resize_height = ceil((float(cv_img.rows)/float(cv_img.cols))*resize_width);
      } else {
        resize_height = resize_short_side;
        resize_width = ceil((float(cv_img.cols)/float(cv_img.rows))*resize_height);
      }
    } else {
      resize_width = resize_size;
      resize_height = resize_size;
    }

    cv::resize(cv_img, cv_resized_img, cv::Size(resize_width, resize_height));
    cv_img_ptr = &cv_resized_img;
    img_height = cv_resized_img.rows;
    img_width = cv_resized_img.cols;
  }

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
                                                                              << "Specify either 1 mean_value or as many as channels: "
                                                                              << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = *cv_img_ptr;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = (*cv_img_ptr)(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] = (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] = (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
    Blob<Dtype>* transformed_blob) {
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset, data_mean_.cpu_data(),
          input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels)
                                                                                << "Specify either 1 mean_value or as many as channels: "
                                                                                << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
              input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w - w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO)<< "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformImgAndSegUniformSize(const Datum& datum,
    Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label) {
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int channels = transformed_blob->channels();
  const int blob_height = transformed_blob->height();
  const int blob_width = transformed_blob->width();
  const int num = transformed_blob->num();

  if (transformed_label) {
    CHECK_EQ(blob_height, transformed_label->shape(2));
    CHECK_EQ(blob_width, transformed_label->shape(3));
  }

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(blob_height, datum_height);
  CHECK_LE(blob_width, datum_width);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();
  int crop_height = param_.crop_height();
  int crop_width = param_.crop_width();
  if (crop_size > 0) {
    CHECK_EQ(crop_height, 0);
    CHECK_EQ(crop_width, 0);
    crop_height = crop_size;
    crop_width = crop_size;
  }

  if (crop_height || crop_width) {
    CHECK_GT(crop_height, 0);
    CHECK_GT(crop_width, 0);
    CHECK_EQ(crop_height, blob_height);
    CHECK_EQ(crop_width, blob_width);
  } else {
    CHECK_EQ(datum_height, blob_height);
    CHECK_EQ(datum_width, blob_width);
  }

  const string& data = datum.data();

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
                                                                                << "Specify either 1 mean_value or as many as channels: "
                                                                                << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_height || crop_width) {
    CHECK_GT(crop_height, 0);
    CHECK_GT(crop_width, 0);
    height = crop_height;
    width = crop_width;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_height + 1);
      w_off = Rand(datum_width - crop_width + 1);
    } else {
      h_off = (datum_height - crop_height) / 2;
      w_off = (datum_width - crop_width) / 2;
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Dtype* transformed_label_data = NULL;
  if (transformed_label) {
    transformed_label_data = transformed_label->mutable_cpu_data();
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (transformed_label) {
          transformed_label_data[top_index] = datum.labels(data_index);
        }
        if (has_uint8) {
          datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_values) {
          transformed_data[top_index] = (datum_element - mean_values_[c])
              * scale;
        } else {
          transformed_data[top_index] = datum_element * scale;
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformImgAndSegUniformSize(const Datum& datum,
    const cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
    Blob<Dtype>* transformed_label) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  int crop_height = param_.crop_height();
  int crop_width = param_.crop_width();
  if (crop_size > 0) {
    CHECK_EQ(crop_height, 0);
    CHECK_EQ(crop_width, 0);
    crop_height = crop_size;
    crop_width = crop_size;
  }
  if (crop_height == 0) {
    crop_height = crop_height_from_aspect_ratio_;
    crop_width = crop_width_from_aspect_ratio_;
  }
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_height);
  CHECK_GE(img_width, crop_width);
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
                                                                              << "Specify either 1 mean_value or as many as channels: "
                                                                              << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_height || crop_width) {
    CHECK_GT(crop_height, 0);
    CHECK_GT(crop_width, 0);
    CHECK_EQ(crop_height, height);
    CHECK_EQ(crop_width, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_height + 1);
      w_off = Rand(img_width - crop_width + 1);
    } else {
      h_off = (img_height - crop_height) / 2;
      w_off = (img_width - crop_width) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_width, crop_height);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }
  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Dtype* transformed_label_data = NULL;
  if (transformed_label) {
    transformed_label_data = transformed_label->mutable_cpu_data();
  }
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      if (transformed_label) {
        if (do_mirror) {
          top_index = h * width + (width - 1 - w);
        } else {
          top_index = h * width + w;
        }
        int label_index = (h + h_off) * img_width + w + w_off;
        CHECK_LT(label_index, datum.labels_size());
        transformed_label_data[top_index] = datum.labels(label_index);
      }

      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_values) {
          transformed_data[top_index] = (pixel - mean_values_[c]) * scale;
        } else {
          transformed_data[top_index] = pixel * scale;
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::ComputeCropHeightWidth(Dtype aspect_ratio) {
  if (((Dtype) min_height_ / (Dtype) min_width_) < aspect_ratio) {
    crop_height_from_aspect_ratio_ = min_height_;
    crop_width_from_aspect_ratio_ = floor(min_height_ / aspect_ratio);
  } else {
    crop_width_from_aspect_ratio_ = min_width_;
    crop_height_from_aspect_ratio_ = floor(min_width_ * aspect_ratio);
  }
  crop_height_from_aspect_ratio_ = (crop_height_from_aspect_ratio_
      / param_.height_multiple()) * param_.height_multiple();
  crop_width_from_aspect_ratio_ = (crop_width_from_aspect_ratio_
      / param_.width_multiple()) * param_.width_multiple();
  CHECK_GT(crop_height_from_aspect_ratio_, 0);
  CHECK_GT(crop_width_from_aspect_ratio_, 0);
  CHECK_LE(crop_height_from_aspect_ratio_, min_height_);
  CHECK_LE(crop_width_from_aspect_ratio_, min_width_);

  DLOG(INFO)<<"aspect_ratio, cropping height and width: "
  <<aspect_ratio
  <<" "<<crop_height_from_aspect_ratio_
  <<" "<<crop_width_from_aspect_ratio_;
}

template<typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror()
      || (phase_ == TRAIN && param_.crop_size()) ||
      (param_.resize_short_side_min() > 0 && param_.resize_short_side_max() > 0);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template<typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
