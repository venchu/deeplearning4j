/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.convolution;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import static jcuda.jcublas.JCublas2.cublasCreate;

import java.util.Arrays;


/**
 * Convolution layer integrated to CuDNN
 *
 */
public class ConvolutionCuDNNLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected INDArray col; // vectorized input

    public ConvolutionCuDNNLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ConvolutionCuDNNLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    public double calcL2() {
        // TODO keep params on gpu and pass in l2 norm to apply
    	if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;

        double l2Norm = getParam(ConvolutionParamInitializer.WEIGHT_KEY).norm2Number().doubleValue();
        return 0.5 * conf.getLayer().getL2() * l2Norm * l2Norm;
    }

    @Override
    public double calcL1() {
        // TODO keep params on gpu and pass in l1 norm to apply

    	if(!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0 ) return 0.0;
        return conf.getLayer().getL1() * getParam(ConvolutionParamInitializer.WEIGHT_KEY).norm1Number().doubleValue();
    }


    public INDArray calculateDelta(INDArray epsilon) {
        INDArray z = preOutput(true);
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), z).derivative());
        if(!Arrays.equals(z.shape(),activationDerivative.shape()))
            throw new IllegalStateException("Shapes must be same");
        return epsilon.muli(activationDerivative);

    }

    void addBias(cudnnTensorDescriptor dstTensorDesc,
                 Layer layer, int c, Pointer data)
    {
        cudnnSetTensor4dDescriptor(biasTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, c, 1, 1);
        Pointer alpha = pointerTo(1.0f);
        Pointer beta = pointerTo(1.0f);
        cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C, alpha,
                biasTensorDesc, layer.bias_d, beta, dstTensorDesc, data);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);

        // gy, Note: epsilon should be reshaped to a tensor when passed in
        INDArray delta = calculateDelta(epsilon);

        Gradient retGradient = new DefaultGradient();

        //gb = gy[0].sum(axis=(0, 2, 3))
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, delta.sum(0, 2, 3));

        // gW = np.tensordot(gy[0], col, ([0, 2, 3], [0, 4, 5]))
        INDArray weightGradient = Nd4j.tensorMmul(delta, col, new int[][] {{0, 2, 3},{0, 4, 5}});
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradient);

        //gcol = tensorMmul(W, gy[0], (0, 1))
        INDArray nextEpsilon = Nd4j.tensorMmul(weights, delta, new int[][] {{0}, {1}});

        nextEpsilon = Nd4j.rollAxis(nextEpsilon, 3);
        nextEpsilon = Convolution.col2im(nextEpsilon, layerConf().getStride(), layerConf().getPadding(), inputHeight, inputWidth);
        return new Pair<>(retGradient,nextEpsilon);
    }

    public INDArray preOutput(boolean training) {
        INDArray Weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(ConvolutionParamInitializer.BIAS_KEY);
        if(conf.isUseDropConnect() && training) {
            if (conf.getLayer().getDropOut() > 0) {
                Weights = Dropout.applyDropConnect(this, ConvolutionParamInitializer.WEIGHT_KEY);
            }
        }

        INDArray z = Nd4j.tensorMmul(col, Weights, new int[][]{{1, 2, 3}, {1, 2, 3}});
        BroadcastOp op = new BroadcastAddOp(z,bias,z,3);
        Nd4j.getExecutioner().exec(op);

        return Nd4j.rollAxis(z, 3, 1);
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");
        applyDropOutIfNecessary(training);

        col = Convolution.im2col(input, layerConf().getKernelSize(), layerConf().getStride(), layerConf().getPadding());
        INDArray z = preOutput(training);
        // TODO add switch here to use bn if included
        INDArray activation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), z));
        return activation;
    }

    void convoluteForward(Layer conv, TensorLayout t,
                          Pointer srcData, Pointer dstData)
    {
        int algo = 0; // cudnnConvolutionFwdAlgo_t

        cudnnSetTensor4dDescriptor(srcTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);

        int tensorDims = 4;
        int tensorOuputDimA[] = { t.n, t.c, t.h, t.w };
        int filterDimA[] = {
                conv.outputs, conv.inputs,
                conv.kernel_dim, conv.kernel_dim };
        cudnnSetFilterNdDescriptor(filterDesc,
                CUDNN_DATA_FLOAT, tensorDims, filterDimA);

        int convDims = 2;
        int padA[] = { 0, 0 };
        int filterStrideA[] = { 1, 1 };
        int upscaleA[] = { 1, 1 };
        cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA,
                filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION);

        // find dimension of convolution output
        cudnnGetConvolutionNdForwardOutputDim(convDesc,
                srcTensorDesc, filterDesc,
                tensorDims, tensorOuputDimA);
        t.n = tensorOuputDimA[0];
        t.c = tensorOuputDimA[1];
        t.h = tensorOuputDimA[2];
        t.w = tensorOuputDimA[3];

        cudnnSetTensor4dDescriptor(dstTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);

        if (convAlgorithm < 0)
        {
            int algoArray[] = { -1 };

            // Choose the best according to the preference
            System.out.println(
                    "Testing cudnnGetConvolutionForwardAlgorithm ...");
            cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc,
                    filterDesc, convDesc, dstTensorDesc,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
            algo = algoArray[0];

            System.out.println("Fastest algorithm is Algo " + algo);
            convAlgorithm = algo;

            // New way of finding the fastest config
            // Setup for findFastest call
            System.out.println(
                    "Testing cudnnFindConvolutionForwardAlgorithm ...");
            int requestedAlgoCount = 5;
            int returnedAlgoCount[] = new int[1];
            cudnnConvolutionFwdAlgoPerf results[] =
                    new cudnnConvolutionFwdAlgoPerf[requestedAlgoCount];
            cudnnFindConvolutionForwardAlgorithm(cudnnHandle,
                    srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                    requestedAlgoCount, returnedAlgoCount, results);
            for (int algoIndex = 0; algoIndex < returnedAlgoCount[0]; ++algoIndex)
            {
                System.out.printf(
                        "    %s for Algo %d (%s): %f time requiring %d memory\n",
                        cudnnGetErrorString(results[algoIndex].status),
                        results[algoIndex].algo,
                        cudnnConvolutionFwdAlgo.stringFor(results[algoIndex].algo),
                        results[algoIndex].time, results[algoIndex].memory);
            }
        }
        else
        {
            algo = convAlgorithm;
            if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            {
                System.out.println("Using FFT for convolution");
            }
        }

        resize(t.n * t.c * t.h * t.w, dstData);
        long sizeInBytesArray[] = { 0 };
        Pointer workSpace = new Pointer();
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                algo, sizeInBytesArray);
        long sizeInBytes = sizeInBytesArray[0];
        if (sizeInBytes != 0)
        {
            cudaMalloc(workSpace, sizeInBytes);
        }

        Pointer alpha = pointerTo(1.0f);
        Pointer beta = pointerTo(0.0f);
        cudnnConvolutionForward(cudnnHandle, alpha, srcTensorDesc,
                srcData, filterDesc, conv.data_d, convDesc, algo,
                workSpace, sizeInBytes, beta, dstTensorDesc, dstData);
        addBias(dstTensorDesc, conv, t.c, dstData);
        if (sizeInBytes != 0)
        {
            cudaFree(workSpace);
        }
    }
    void activationForward(TensorLayout t,
                           Pointer srcData, Pointer dstData)
    {
        resize(t.n * t.c * t.h * t.w, dstData);

        cudnnSetTensor4dDescriptor(srcTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
        cudnnSetTensor4dDescriptor(dstTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);

        Pointer alpha = pointerTo(1.0f);
        Pointer beta = pointerTo(0.0f);
        cudnnActivationForward(cudnnHandle, CUDNN_ACTIVATION_RELU,
                alpha, srcTensorDesc, srcData,
                beta, dstTensorDesc, dstData);
    }


}
