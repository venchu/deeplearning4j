package org.deeplearning4j.cli.conf.examples.dbn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.conf.rng.DefaultRandom;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Known good configs saved in code to generate json network architectures fpr DBNs
 * 
 * @author josh
 *
 */
public class ConfigGenerator {


	public static void main(String [ ] args)
	{
		generate_Iris_ModelArchitecture("/tmp/generated_dbn_iris_arch.json");
	}
	
	
	/**
	 * Based on:
	 * 
	 * 	https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNIrisExample.java
	 * 
	 * 
	 * @param modelConfPath
	 */
	public static void generate_Iris_ModelArchitecture(String modelConfPath) {
		

        final int numRows = 4;
        final int numColumns = 1;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 5;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = iterations-1;		
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .layer(new RBM()) // NN layer type
        .nIn(numRows * numColumns) // # input nodes
        .nOut(outputNum) // # output nodes
        .seed(seed) // Seed to lock in weight initialization for tuning
        .visibleUnit(RBM.VisibleUnit.GAUSSIAN) // Gaussian transformation visible layer
        .hiddenUnit(RBM.HiddenUnit.RECTIFIED) // Rectified Linear transformation visible layer
        .iterations(iterations) // # training iterations predict/classify & backprop
        .weightInit(WeightInit.XAVIER) // Weight initialization method
        .activationFunction("relu") // Activation function type
        .k(1) // # contrastive divergence iterations
        .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
        .learningRate(1e-3f) // Optimization step size
        .optimizationAlgo(OptimizationAlgorithm.LBFGS) // Backprop method (calculate the gradients)
        .constrainGradientToUnitNorm(true)
        .useDropConnect(true)
        .regularization(true)
        .l2(2e-4)
        .momentum(0.9)
        .list(2) // # NN layers (does not count input layer)
        .hiddenLayerSizes(9) // # fully connected hidden layer nodes. Add list if multiple layers.
        .override(1, new ConfOverride() {
            @Override
            public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                builder.activationFunction("softmax");
                builder.layer(new OutputLayer());
                builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                builder.optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT);
            }
        })
        .build();	
		
		

        System.out.println( "DBN Conf: \n" );
        String c = conf.toJson();
        System.out.println(c);
        
        File file = new File( modelConfPath );
        
        if (!file.exists()){
        	System.out.println("creating file: " + modelConfPath );
           try {
			file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        } else {
        	System.out.println( "already exists" );
        }
        
        PrintWriter out = null;
        
        try {
			out = new PrintWriter( file );
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        out.println(c);
        out.close();
        				
		
	}

}
