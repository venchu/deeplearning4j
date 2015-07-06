package org.deeplearning4j.cli.subcommands;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Properties;

import org.deeplearning4j.cli.conf.ModelConfigurationUtil;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Generate implements SubCommand {
	
	private static Logger log = LoggerFactory.getLogger(Generate.class);

    public static final String NETWORK_ARCHITECTURE_KEY = "dl4j.model.config.architecture";
    public static final String NETWORK_ARCHITECTURE_DEFAULT = "dbn";

    @Option(name = "-conf", usage = "configuration file for generating the network architecture" )
    public String configurationFile = "";
    
    public boolean validCommandLineParameters = false;
    public boolean validModelConfigJSONFile = false;
    public Properties configProps = null;
    public String modelConfigPath = "";
    public String networkArchitecture = NETWORK_ARCHITECTURE_DEFAULT;
    protected String[] args;
    
    public Generate() { 
    	
  //  	this({ "" });
    };
    
    public Generate(String[] args) {
        
    	
    	/*
    	super(args);
        
    	System.out.println("Generate CTOR");
        
        
        this.args = args;
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            this.validCommandLineParameters = false;
            //parser.printUsage(System.err);
            //log.error("Unable to parse args", e);
            printUsage();
        }
        */
    	

        this.args = args;
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
        	
        	System.out.println( "early err -----" );
        	
            this.validCommandLineParameters = false;
            parser.printUsage(System.err);
            log.error("Unable to parse args", e);
        }
    	
        
    }
    
    
    
    public static void printUsage() {
    	
    	System.out.println( "DL4J: CLI Model Achitecture JSON Generator" );
    	System.out.println( "" );
    	System.out.println( "\tUsage:" );
    	System.out.println( "\t\tdl4j generate -conf <conf_file>" );
    	System.out.println( "" );
    	System.out.println( "\tConfiguration File:" );
    	System.out.println( "\t\tContains a list of property entries that describe the model configuration generation process" );
    	System.out.println( "" );
    	System.out.println( "\tExample:" );
    	System.out.println( "\t\tdl4j generate -conf /tmp/iris_conf.txt " );
    	
    	
    }  
    
    @Override
    public void execute() {
    	
    	
    	if ("".equals(this.configurationFile)) {
    		printUsage();
    		return;
    	}

    
        try {
            loadConfigFile();
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        ModelConfigurationUtil confUtil = new ModelConfigurationUtil( this.configProps );
        
        
        if ("dbn".equals(this.networkArchitecture.trim())) {

        	confUtil.generateDefaultDBNConfigFile();
        	
        } else {
        	
        	System.out.println( "Error: Network Architecture Unrecognized / Unsupported: " + this.networkArchitecture );
        	
        }



    }  
    
    public void loadConfigFile() throws Exception {

        this.configProps = new Properties();
        
        
//System.out.println( "loading conf file: " + this.configurationFile );
        InputStream in = null;
        try {
            in = new FileInputStream( this.configurationFile );
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        try {
            this.configProps.load(in);
            in.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }


/*
        // get runtime - EXECUTION_RUNTIME_MODE_KEY
        if (this.configProps.get( Train.EXECUTION_RUNTIME_MODE_KEY ) != null) {
            this.runtime = (String) this.configProps.get(Train.EXECUTION_RUNTIME_MODE_KEY);

        } else {
            this.runtime = Train.EXECUTION_RUNTIME_MODE_DEFAULT;
        }
*/
        /*
        // get output directory
        if (null != this.configProps.get( Train.OUTPUT_FILENAME_KEY )) {
        	
        
            this.outputDirectory = (String) this.configProps.get(OUTPUT_FILENAME_KEY);

        } else {
            // default
            this.outputDirectory = "/tmp/dl4_model_default.model";
        //throw new Exception("no output location!");
        }
*/
/*
        // get input data

        if ( null != this.configProps.get( INPUT_DATA_FILENAME_KEY )) {
        	
            this.input = (String) this.configProps.get(INPUT_DATA_FILENAME_KEY);

        } else {
            throw new RuntimeException("no input file to train on!");
        }
        */

        // get MODEL_CONFIG_KEY
        
        if ( null != this.configProps.get(Train.MODEL_CONFIG_KEY)) {
        	
        	this.modelConfigPath = (String) this.configProps.getProperty(Train.MODEL_CONFIG_KEY);
        	
        	System.out.println( "Working json path: " + this.modelConfigPath );
        	
        } else {
        	
        	// need to auto-gen the model config JSON file [ cold start problem ]
        	
        	System.out.println( "Warning: No model path was defined; We don't know where to save your model configuration!" );
        	
        }
        
        // NETWORK_ARCHITECTURE_KEY
        if ( null != this.configProps.get( NETWORK_ARCHITECTURE_KEY )) {
        	
        	this.networkArchitecture = (String) this.configProps.getProperty( NETWORK_ARCHITECTURE_KEY );
        	
        } else {
        	
        	// need to auto-gen the model config JSON file [ cold start problem ]
        	
        	System.out.println( "Warning: No network architecture was defined!" );
        	
        }
        
        System.out.println( "Generating network architecture json file at: " + this.modelConfigPath );
        
    }    
    
    
}
