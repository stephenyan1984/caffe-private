#include <fstream>  // NOLINT(readability/streams)
#include <algorithm>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

using namespace caffe;
using boost::scoped_ptr;

DEFINE_bool(shuffle, false, "Randomly shuffle the order of images");
DEFINE_int32(min_height, 1, "minimal image height");
DEFINE_int32(min_width, 1, "minimal image width");

DEFINE_string(image_dir, "", "The folder where images reside") ;
DEFINE_string(label_dir, "",
    "The folder where groundtruth segmentation label files reside") ;
DEFINE_string(image_list_file, "", "Text file listing the images") ;
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(out_database_path, "", "The path of output database") ;

DEFINE_bool(image_name_as_key, false,
    "If false, use image name in the list as the key"
    "Otherwise, use ('%d' % image_index) as key");

void ResizeLabelMap(const vector<vector<int> >& original_label_map,
    vector<vector<int> >& label_map, const int height, const int width);

void ReadLabelFile(const caffe::string& label_file_name, int img_height,
    int img_width, vector<vector<int> >& label_map);

bool ReadImageSegToDatum(const caffe::string& img_name,
    const caffe::string& label_file_name, const int min_height,
    const int min_width, Datum *datum);

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

  gflags::SetUsageMessage(
      "Convert a list of images and their pixel annotation into a leveldb/lmdb \n"
          "format used as input for Caffe.\n"
          "Usage:\n"
          "    convert_stanford_background_data [FLAGS] IMAGE_DIR/ LABEL_DIR/ LIST_FILE DB_NAME\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_image_list_file != std::string(""))
                                                     << "Image list file is not given";

  std::string img_name;
  std::vector<std::string> img_names;
  std::ifstream img_list_file(FLAGS_image_list_file.c_str());
  while (img_list_file >> img_name) {
    img_names.push_back(img_name);
  }
  img_list_file.close();

  if (FLAGS_shuffle) {
    LOG(INFO)<<"shuffle images in the list";
    shuffle(img_names.begin(), img_names.end());
  } else {
    LOG(INFO)<<"Do not shuffle images in the list";
  }

  if(FLAGS_image_name_as_key) {
    LOG(INFO)<<"Use image name in the list as the key";
  } else {
    LOG(INFO)<<"Use integer image index in the list as the key";
  }

  // create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(FLAGS_out_database_path, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Store into DB
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  for (int i = 0; i < img_names.size(); ++i) {
    Datum datum;
    std::string img_path = FLAGS_image_dir + img_names[i] + std::string(".jpg");
    std::string label_path = FLAGS_label_dir + img_names[i]
        + std::string(".txt");
    bool status = ReadImageSegToDatum(img_path, label_path, FLAGS_min_height,
        FLAGS_min_width, &datum);
    if (!status) {
      continue;
    }
    int length = 0;
    if (FLAGS_image_name_as_key) {
      length = snprintf(key_cstr, kMaxKeyLength, "%s", img_names[i].c_str());
    } else {
      length = snprintf(key_cstr, kMaxKeyLength, "%08d", count);
    }

    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);
    if (++count % 200 == 0) {
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR)<<"Processed "<<count<<" images";
    }
  }
  if (count % 200 != 0) {
    txn->Commit();
    LOG(ERROR)<<"Processed "<<count<<" images";
  }
  return 0;
}

//  resize label map using simple nearest neighbor interpolation
void ResizeLabelMap(const vector<vector<int> >& original_label_map,
    vector<vector<int> >& label_map, const int height, const int width) {
  const int original_height = original_label_map.size();
  const int original_width = original_label_map[0].size();

  label_map.resize(height);
  for (int i = 0; i < height; ++i) {
    label_map[i].resize(width);
    float fy = (float) i / (float) (height - 1);
    int ry = floor(fy * (float) (original_height - 1) + 0.5);
    for (int j = 0; j < width; ++j) {
      float fx = (float) j / (float) (width - 1);
      int rx = floor(fx * (float) (original_width - 1) + 0.5);
      label_map[i][j] = original_label_map[ry][rx];
    }
  }
}

void ReadLabelFile(const caffe::string& label_file_name, int img_height,
    int img_width, vector<vector<int> >& label_map) {
  std::ifstream infile(label_file_name.c_str());
  caffe::string line;
  CHECK(infile.is_open()) << "Can NOT open file " << label_file_name;

  label_map.resize(img_height);
  int row = 0;
  while (getline(infile, line)) {
    vector<std::string> row_labels;
    boost::split(row_labels, line, boost::is_any_of(" "));
    CHECK_EQ(row_labels.size(), img_width);
    label_map[row].resize(img_width);
    for (int i = 0; i < img_width; ++i) {
      int lb = atoi(row_labels[i].c_str());
      if (lb == 255) {
        // label 255 denotes "difficult" pixels
        // relabel it to 21.
        // label "0" denotes background label
        // labes 1 ~ 20 denote foreground object labels
        lb = 21;
      }
      label_map[row][i] = lb;
    }
    row++;
  }
  CHECK_EQ(row, img_height);
  infile.close();
}

bool ReadImageSegToDatum(const caffe::string& img_name,
    const caffe::string& label_file_name, const int min_height,
    const int min_width, Datum *datum) {
  cv::Mat cv_original_img = ReadImageToCVMat(img_name, 0, 0, true);
  const int img_height = cv_original_img.rows;
  const int img_width = cv_original_img.cols;

  cv::Mat cv_img;
  vector<vector<int> > label_map;

  int new_height = 0, new_width = 0;
  if (img_height < min_height || img_width < min_width) {
    // enlarge image to meet the minimal height/width requirement
    vector<vector<int> > original_label_map;
    ReadLabelFile(label_file_name, img_height, img_width, original_label_map);

    // may need to resize the image so that it meets the requirements
    // of minimum height and width with aspect ratio preserved
    if (img_height < min_height && img_width >= min_width) {
      double scaling_f = (double) min_height / (double) img_height;
      new_height = min_height;
      new_width = ceil(img_width * scaling_f);
    } else if (img_height >= min_height && img_width < min_width) {
      double scaling_f = (double) min_width / (double) img_width;
      new_height = ceil(img_height * scaling_f);
      new_width = min_width;
    } else {
      double scaling_height_f = (double) min_height / (double) img_height;
      double scaling_width_f = (double) min_width / (double) img_width;
      bool scaling_height = scaling_height_f <= scaling_width_f ? false : true;
      new_height =
          scaling_height ? min_height : ceil(img_height * scaling_width_f);
      new_width =
          scaling_height ? ceil(img_width * scaling_height_f) : min_width;
    }
    cv::resize(cv_original_img, cv_img, cv::Size(new_width, new_height));
    ResizeLabelMap(original_label_map, label_map, new_height, new_width);
  } else {
    new_height = img_height;
    new_width = img_width;
    cv_img = cv_original_img;
    ReadLabelFile(label_file_name, img_height, img_width, label_map);
  }
  DLOG(INFO)<<"ReadImageSegToDatum resize image from ("
  <<img_height<<","<<img_width<<") to ("
  <<new_height<<","<<new_width<<")";
  CHECK_GE(new_width, min_width);
  CHECK_GE(new_height, min_height);

  vector<uchar> buf;
  cv::imencode(".jpg", cv_img, buf);
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_data(reinterpret_cast<char*>(&buf[0]), buf.size());
  datum->set_encoded(true);
  datum->clear_labels();

  for (int y = 0; y < cv_img.rows; ++y) {
    for (int x = 0; x < cv_img.cols; ++x) {
      datum->add_labels(label_map[y][x]);
    }
  }
  return true;
}
