#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  LOG_FIRST_N(INFO, 10) << "Thread joined";
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    top[1]->Reshape(prefetch_label_.num(), prefetch_label_.channels(),
        prefetch_label_.height(), prefetch_label_.width());
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  LOG_FIRST_N(INFO, 10) << "Prefetch copied";

  if (top.size() == 3) {
    Dtype *size_blob_data = top[2]->mutable_cpu_data();
    size_blob_data[0] = this->prefetch_data_.height();
    size_blob_data[1] = this->prefetch_data_.width();
    LOG_FIRST_N(INFO, 2)<<"ImageSegUniformSizeDataLayer<Dtype>::Forward_cpu height "
        <<size_blob_data[0]<<" width "<<size_blob_data[1];
  }
  // Start a new prefetch thread
  LOG_FIRST_N(INFO, 10) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
