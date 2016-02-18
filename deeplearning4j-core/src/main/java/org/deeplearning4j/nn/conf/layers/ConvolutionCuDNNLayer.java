package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.nd4j.linalg.convolution.Convolution;

/**
 */

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ConvolutionCuDNNLayer extends ConvolutionLayer {

    protected ConvolutionCuDNNLayer(Builder builder){
        super(builder);
    }

    @Override
    public ConvolutionCuDNNLayer clone() {
        ConvolutionCuDNNLayer clone = (ConvolutionCuDNNLayer) super.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends ConvolutionLayer.Builder<Builder> {

        @Override
        @SuppressWarnings("unchecked")
        public ConvolutionCuDNNLayer build() {
            return new ConvolutionCuDNNLayer(this);
        }
    }
}
