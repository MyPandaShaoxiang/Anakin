#include "saber/funcs/impl/x86/saber_pad.h"
#include "saber/core/tensor_op.h"

namespace anakin{
namespace saber{

template <>
SaberStatus SaberPad<X86, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<X86>*>& inputs,
                std::vector<Tensor<X86>*>& outputs,
                PadParam<X86> &param){
        if (! inputs[0]->is_continue_mem() || ! outputs[0]->is_continue_mem()){
            LOG(ERROR)<<"pad only support tensor with continue memory";
            return SaberUnImplError;
        }
        const float* src_ptr = static_cast<const float*>(inputs[0]->data());
        float* dst_ptr = static_cast<float*>(outputs[0]->mutable_data());
        
        int in_n = inputs[0] -> num();
        int in_c = inputs[0] -> channel();
        int in_h = inputs[0] -> height();
        int in_w = inputs[0] -> width();
        int out_n = outputs[0] -> num();
        int out_c = outputs[0] -> channel();
        int out_h = outputs[0] -> height();
        int out_w = outputs[0] -> width();
        Shape in_stride = inputs[0] -> get_stride();
        Shape out_stride = outputs[0] -> get_stride();
        int in_idn = inputs[0] -> num_index();
        int in_idc = inputs[0] -> channel_index();
        int in_idh = inputs[0] -> height_index();
        int in_idw = inputs[0] -> width_index();
        int out_idn = outputs[0] -> num_index();
        int out_idc = outputs[0] -> channel_index();
        int out_idh = outputs[0] -> height_index();
        int out_idw = outputs[0] -> width_index();
        
        fill_tensor_const(*outputs[0], 0);
        
        int c0 = param.pad_c[0];
        int h0 = param.pad_h[0];
        int w0 = param.pad_w[0];
        int offset = c0 * out_stride[out_idc] + h0 * out_stride[out_idh] + w0 * out_stride[out_idw];
        for (int id = 0; id < inputs[0] -> valid_size(); ++id){
            int i_n = (id / in_stride[in_idn]) % in_n;
            int i_c = (id / in_stride[in_idc]) % in_c;
            int i_h = (id / in_stride[in_idh]) % in_h;
            int i_w = (id / in_stride[in_idw]) % in_w;
            int out_id = i_n * out_stride[out_idn] + i_c * out_stride[out_idc] + i_h * out_stride[out_idh] + i_w * out_stride[out_idw];
            dst_ptr[out_id + offset] = src_ptr[id];
        }
}
    
}
}
