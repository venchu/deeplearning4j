package org.deeplearning4j.cli.conf.examples.cnn;

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

public class ConfigGenerator {
	

	public static void main(String [ ] args)
	{
		generateMNISTModelArchitecture("/tmp/stock_cnn_mnist_arch.json");
	}
	
	public static void generateMNISTModelArchitecture(String modelConfPath) {
		
		
		// need to think through default mechanics more
		// TODO: do we update the main conf file with the new property line?
		//String modelConfPath = ""; //this.getPropertyAsString(Train.MODEL_CONFIG_KEY, "/tmp/cnn_model_architecture.json");
        	
       	System.out.println( "Working json path: " + modelConfPath );
		
        int outputNeuronCount = 10; 
		

        int numberInputRows = 28; 
        int numberInputCols = 28; 
        
        int inputNeuronCount = numberInputRows * numberInputCols;
        
//        @Option(name="-samples", usage="number of samples to get")
        int numSamples = 100;

//        @Option(name="-batch", usage="batch size for training" )
        int batchSize = 100;

        //@Option(name="-iterations", usage="number of iterations to train the layer")
        int iterations = 10;

        //@Option(name="-featureMap", usage="size of feature map. Just enter single value")
        final int featureMapSize = 5; 

        //@Option(name="-learningRate", usage="learning rate")
        double learningRate = 0.13;

        //@Option(name="-hLayerSize", usage="hidden layer size")
        int hLayerSize = 18;

        int splitTrainNum = (int) (numSamples * 0.8);
        int numTestSamples = numSamples - splitTrainNum;
        int listenerFreq = iterations/5;        
        
		
		 MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
         .nIn( inputNeuronCount )
         .nOut( outputNeuronCount )
         .batchSize(batchSize)
         .iterations(iterations)
         .weightInit(WeightInit.ZERO)
         .seed(3)
         .activationFunction("sigmoid")
         .filterSize(8, 1, numberInputRows, numberInputCols )
         .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
         .learningRate(learningRate)
         .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
         .constrainGradientToUnitNorm(true)
         .list(3)
         .hiddenLayerSizes(hLayerSize)
         .inputPreProcessor(0, new ConvolutionInputPreProcessor( numberInputRows, numberInputCols ))
         .preProcessor(1, new ConvolutionPostProcessor())
         .override(0, new ConfOverride() {
             public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                 builder.layer(new ConvolutionLayer());
                 builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                 builder.featureMapSize(featureMapSize, featureMapSize);
             }
         })
         .override(1, new ConfOverride() {
             public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                 builder.layer(new SubsamplingLayer());
             }
         })
         .override(2, new ClassifierOverride())
         .build();		
		
        System.out.println( "CNN Conf: \n" );
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
