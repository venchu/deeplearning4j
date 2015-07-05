package org.deeplearning4j.cli.conf;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.Properties;



import org.deeplearning4j.cli.subcommands.Train;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
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
	
    public static final String DBN_MODEL_INPUT_SIZE_KEY = "dl4j.model.dbn.input.size";
    public static final String DBN_MODEL_INPUT_SIZE_DEFAULT = "784";
	
    public static final String DBN_MODEL_OUTPUT_SIZE_KEY = "dl4j.model.dbn.output.size";
    public static final String DBN_MODEL_OUTPUT_SIZE_DEFAULT = "10";
	
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
	 * @param jsonFilePath
	 */
	public void generateDefaultDBNConfigFile() {
		
		String modelConfPath = "";
		
        if ( null != this.configProps.get(Train.MODEL_CONFIG_KEY)) {
        	
        	modelConfPath = (String) this.configProps.getProperty(Train.MODEL_CONFIG_KEY);
        	
        	System.out.println( "Working json path: " + modelConfPath );
        	
        } else {
        	
        	// need to auto-gen the model config JSON file [ cold start problem ]
        	
        	System.out.println( "Warning: No model path was defined; We don't know where to save your model configuration!" );
        	
        }		
		
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
	
	public int calculateInputSizeFromData() {
		
		
		return 0;
	}
	
	public boolean validateExistingJsonConfigFile( String path ) {
		
		
		
		return false;
	}
	

}
