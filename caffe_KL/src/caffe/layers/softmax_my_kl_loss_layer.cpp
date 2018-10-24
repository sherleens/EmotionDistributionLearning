#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_my_kl_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define pi 3.14159265358979323846 //pi------------------------------------------
#define ee 0
namespace caffe {

template <typename Dtype>
void SoftmaxWithMyKLLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithMyKLLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  /*CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";*/
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}
template <typename Dtype>
Dtype SoftmaxWithMyKLLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithMyKLLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  //-----------------------my kl--------------------------------
  //生成高斯分布
  /*
  int kl_temp = 0; 
  double sigma = 0.45, mu = 0;
  double m_pro[8][8], m_sum = 0;
  for (int i = 0; i < 4; i++){
      m_sum = 0;
      for (int j = 0; j < 8; j++){
          mu = i;
          m_pro[i][j] = 1.0 / (sqrt(2 * pi)*sigma) * exp(-1 * (j - mu)*(j - mu) / (2 * sigma*sigma));
          if (j > 3)
              m_pro[i][j] = ee;
          m_sum += m_pro[i][j];
      }
      for (int j = 0; j< 8; j++){
          m_pro[i][j] = m_pro[i][j] / m_sum;
      }
  }
  for (int i = 4; i < 8; i++){
      m_sum = 0;
      for (int j = 0; j < 8; j++){
          mu = i;
          m_pro[i][j] = 1.0 / (sqrt(2 * pi)*sigma) * exp(-1 * (j - mu)*(j - mu) / (2 * sigma*sigma));
          if (j < 4)
              m_pro[i][j] = ee;
          m_sum += m_pro[i][j];
      }
      for (int j = 0; j< 8; j++){
          m_pro[i][j] = m_pro[i][j] / m_sum;
      }
  }*/
  //---------------------------------------------------------------
  
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      //const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      //if (has_ignore_label_ && label_value == ignore_label_) {
      //  continue;
      //}
      //DCHECK_GE(label_value, 0);
      //DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      //loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
      //                     Dtype(FLT_MIN)));
      //-----------------------my kl--------------------------------
  
      float kl_temp = 0;
      float ismax=0;
      int nn = 0;
      for (int k = 0; k < dim; k++) {
          //kl_temp += label[i * dim + k * inner_num_ + j] * log(prob_data[i * dim + k * inner_num_ + j]);
          if (label[i * dim + k * inner_num_ + j] > ismax) {
              nn = k;
              ismax = label[i * dim + k * inner_num_ + j];
          }
      }
      DCHECK_GE(nn, 0);
      DCHECK_LT(nn, prob_.shape(softmax_axis_));
      
      kl_temp += log(std::max(prob_data[i * dim + nn * inner_num_ + j],Dtype(FLT_MIN)));
      
      /*if (label_value<4){
          for (int k = 0; k < 4; k++) {
              kl_temp +=  m_pro[label_value][k] * log(prob_data[i * dim + k * inner_num_ + j]);
          }
      }
      else {      
          for (int k = 4; k < dim; k++) {
              kl_temp +=  m_pro[label_value][k] * log(prob_data[i * dim + k * inner_num_ + j]);    
        }
      } */
      
     // for (int k = 0; k < dim; k++) {
     //     kl_temp +=  m_pro[label_value][k] * log(prob_data[i * dim + k * inner_num_ + j]);
      //}
      
      loss -= (kl_temp);
      //---------------------------------------------------------------
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithMyKLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    //-----------------------my kl--------------------------------
    //生成高斯分布
    /*double sigma = 0.45, mu = 0;
    double m_pro[8][8], m_sum = 0;
    for (int i = 0; i < 4; i++){
        m_sum = 0;
        for (int j = 0; j < 8; j++){
            mu = i;
            m_pro[i][j] = 1.0 / (sqrt(2 * pi)*sigma) * exp(-1 * (j - mu)*(j - mu) / (2 * sigma*sigma));
            if (j > 3)
                m_pro[i][j] =ee;
            m_sum += m_pro[i][j];
        }
        for (int j = 0; j< 8; j++){
            m_pro[i][j] = m_pro[i][j] / m_sum;
        }
    }
    for (int i = 4; i < 8; i++){
        m_sum = 0;
        for (int j = 0; j < 8; j++){
            mu = i;
            m_pro[i][j] = 1.0 / (sqrt(2 * pi)*sigma) * exp(-1 * (j - mu)*(j - mu) / (2 * sigma*sigma));
            if (j < 4)
                m_pro[i][j] = ee;
            m_sum += m_pro[i][j];
        }
        for (int j = 0; j< 8; j++){
            m_pro[i][j] = m_pro[i][j] / m_sum;
        }
    }
    //---------------------------------------------------------------
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } 
        else {
            //bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            //my kl loss
            for (int k = 0; k < 8; k++){
                bottom_diff[i * dim + k * inner_num_ + j] -= (m_pro[label_value][k]);
            }
            ++count;
        }
      }
    }*/
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        //const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          //if (has_ignore_label_ && label_value == ignore_label_) {
          //for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
          //  bottom_diff[i * dim + c * inner_num_ + j] = 0;
          //}
          //}
          //else {
          //bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          //my kl loss
          /*
           
           * for (int k = 0; k < dim; k++){
           * bottom_diff[i * dim + k * inner_num_ + j] -= label[i * dim + k * inner_num_ + j];//(m_pro[label_value][k]);
           *
           * if (label[i * dim + k * inner_num_ + j]>ismax) {
           * nn = k;
           * ismax = label[i * dim + k * inner_num_ + j];
           * }
           * }
           * //*/
          float ismax=0;
          int nn = 0;
          for (int k = 0; k < dim; k++){
              //bottom_diff[i * dim + k * inner_num_ + j] -= label[i * dim + k * inner_num_ + j];
              if (label[i * dim + k * inner_num_ + j] > ismax) {
                  nn = k;
                  ismax = label[i * dim + k * inner_num_ + j];
              }
          }
          bottom_diff[i * dim + nn * inner_num_ + j] -= 1;
          
          ++count;
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithMyKLLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithMyKLLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithMyKLLoss);

}  // namespace caffe
