package org.deeplearning4j.cli.conf;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Properties;



import org.apache.commons.io.FileUtils;
import org.deeplearning4j.cli.subcommands.Train;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;

import org.deeplearning4j.nn.conf.layers.RBM;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelConfigurationUtil {
	
	// DBNs -----
	
    public static final String DBN_MODEL_INPUT_SIZE_KEY = "dl4j.model.dbn.input.size";
    public static final String DBN_MODEL_INPUT_SIZE_DEFAULT = "784";
	
    public static final String DBN_MODEL_OUTPUT_SIZE_KEY = "dl4j.model.dbn.output.size";
    public static final String DBN_MODEL_OUTPUT_SIZE_DEFAULT = "10";

    // CNNs ----- 
    
    public static final String CNN_MODEL_INPUT_SIZE_KEY = "dl4j.model.cnn.input.size";
    public static final String CNN_MODEL_INPUT_SIZE_DEFAULT = "784";
	
    public static final String CNN_MODEL_OUTPUT_SIZE_KEY = "dl4j.model.cnn.output.size";
    public static final String CNN_MODEL_OUTPUT_SIZE_DEFAULT = "10";

    public static final String CNN_MODEL_INPUT_ROWS_KEY = "dl4j.model.cnn.input.rows";
    public static final String CNN_MODEL_INPUT_ROWS_DEFAULT = "28";

    public static final String CNN_MODEL_INPUT_COLS_KEY = "dl4j.model.cnn.input.cols";
    public static final String CNN_MODEL_INPUT_COLS_DEFAULT = "28";

    public static final String CNN_MODEL_FEATUREMAP_SIZE_KEY = "dl4j.model.cnn.featuremap.size";
    public static final String CNN_MODEL_FEATUREMAP_SIZE_DEFAULT = "5";
    
    
	public Properties configProps = null;
	
	public ModelConfigurationUtil( Properties props ) {
		
		this.configProps = props;
		
	}
	
	public void generateDefaultModelConfigForArchitecture(String architecture, String baseDir) throws Exception {

		if ("dbn".equals( architecture ) ) {
			
			this.generateDefaultDBNConfigFile( );
			
			
		} else {
			
			throw new Exception("Architecture Not Supported: " + architecture);
			
		}
		
		
		
	}
	
	public int getPropertyAsInt( String key, String defaultValue ) {
		
		String ret = "";
		
        if ( null != this.configProps.get( key )) {
        	
        	ret = (String) this.configProps.getProperty( key );
        	
        } else {

        	ret = defaultValue;
        	
        }		
		
        int iRet = 0;
        try {
        	iRet = Integer.parseInt(ret);
        } catch (Exception e) {
        	
        }
        
        
        return iRet;
		
	}	
	
	public String getPropertyAsString( String key, String defaultValue) {
		
		String ret = "";
		
        if ( null != this.configProps.get( key )) {
        	
        	ret = (String) this.configProps.getProperty( key );
        	
        } else {

        	ret = defaultValue;
        	
        }		
		
        return ret;
		
	}
	
	/**
	 * Need as input:
	 * 	-	input neuron count
	 * 	-	output neuron count 
	 * 
	 * 
	 */
	public void generateDefaultDBNConfigFile() {
		
		// need to think through default mechanics more
		// TODO: do we update the main conf file with the new property line?
		String modelConfPath = this.getPropertyAsString(Train.MODEL_CONFIG_KEY, "/tmp/dbn_model_architecture.json");
        	
       	System.out.println( "Working json path: " + modelConfPath );
		
		//final int numRows = 28;
        //final int numColumns = 28;
        int inputNeuronCount = this.getPropertyAsInt(DBN_MODEL_INPUT_SIZE_KEY, DBN_MODEL_INPUT_SIZE_DEFAULT);
        int outputNeuronCount = this.getPropertyAsInt(DBN_MODEL_OUTPUT_SIZE_KEY, DBN_MODEL_OUTPUT_SIZE_DEFAULT);
		
		int seed = 123;
		int iterations = 10;
		 
        //log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn( inputNeuronCount )
                .nOut( outputNeuronCount )
                .weightInit(WeightInit.XAVIER)
                .seed(seed)
                .constrainGradientToUnitNorm(true)
                .iterations(iterations)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .hiddenLayerSizes(new int[]{500, 250, 200})
                .override(3, new ClassifierOverride())
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
        
        
        //FileUtils
		
	}
	
	
	/**
	 * Generate a new default Convolutional Neural Network configuration file
	 * 
	 * 		- Down the road we want Canova to automatically compute/recommend many of these defaults for a new dataset 
	 * 
	 */
	public void generateDefaultCNNConfigFile() {
		
		// need to think through default mechanics more
		// TODO: do we update the main conf file with the new property line?
		String modelConfPath = this.getPropertyAsString(Train.MODEL_CONFIG_KEY, "/tmp/cnn_model_architecture.json");
        	
       	System.out.println( "Working json path: " + modelConfPath );
		
		//final int numRows = 28;
        //final int numColumns = 28;
//        int inputNeuronCount = this.getPropertyAsInt(CNN_MODEL_INPUT_SIZE_KEY, CNN_MODEL_INPUT_SIZE_DEFAULT);
        int outputNeuronCount = this.getPropertyAsInt(CNN_MODEL_OUTPUT_SIZE_KEY, CNN_MODEL_OUTPUT_SIZE_DEFAULT);
		
//		int seed = 123;
//		int iterations = 10;

//        private static final int numRows = 28;
//        private static final int numColumns = 28;
//        private static final int outputNum = 10;

        int numberInputRows = this.getPropertyAsInt(CNN_MODEL_INPUT_ROWS_KEY, CNN_MODEL_INPUT_ROWS_DEFAULT);
        int numberInputCols = this.getPropertyAsInt(CNN_MODEL_INPUT_COLS_KEY, CNN_MODEL_INPUT_COLS_DEFAULT);
        
        int inputNeuronCount = numberInputRows * numberInputCols;
        
//        @Option(name="-samples", usage="number of samples to get")
        int numSamples = 100;

//        @Option(name="-batch", usage="batch size for training" )
        int batchSize = 100;

        //@Option(name="-iterations", usage="number of iterations to train the layer")
        int iterations = 10;

        //@Option(name="-featureMap", usage="size of feature map. Just enter single value")
        final int featureMapSize = this.getPropertyAsInt(CNN_MODEL_FEATUREMAP_SIZE_KEY, CNN_MODEL_FEATUREMAP_SIZE_DEFAULT);

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
        
        
        //FileUtils
		
	}	
	
	public int calculateInputSizeFromData() {
		
		
		return 0;
	}
	
	
	public static boolean validateExistingJsonConfigFile( String path ) {
		
		String content = "";
		try {
			//content = readFile(path, Charset.defaultCharset());
			content = FileUtils.readFileToString( new File( path ) );
		} catch (IOException e) {
			// TODO Auto-generated catch block
//			e.printStackTrace();
			System.out.println( "Could not load " + path + " to validate the model architecture." );
			return false;
		}
		
	//	MultiLayerNetwork model = new MultiLayerNetwork(conf);
     //   model.init();
		
		MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(content);
			MultiLayerNetwork model = new MultiLayerNetwork(conf);
	        model.init();
	        
		
		
		
		return true;
	}
	

}
