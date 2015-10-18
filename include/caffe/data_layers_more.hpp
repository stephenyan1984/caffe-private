#ifndef CAFFE_DATA_LAYERS_MORE_HPP_
#define CAFFE_DATA_LAYERS_MORE_HPP_

#include "caffe/data_layers.hpp"

namespace caffe {
/**
 * @brief Image and segmentation pair data provider.
 * Image sizes are uniform within the mini-batch
 * OUTPUT:
 * 0: (num, channels, height, width): image values
 * 1: (num, 1, height, width): labels
 */
template <typename Dtype>
class ImageSegUniformSizeDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegUniformSizeDataLayer(const LayerParameter& param);
  virtual ~ImageSegUniformSizeDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSegUniformSizeData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }
 protected:
  virtual void InternalThreadEntry();

  Blob<Dtype> transformed_label_;
  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
};

} // namespace caffe

#endif // CAFFE_DATA_LAYERS_MORE_HPP_
