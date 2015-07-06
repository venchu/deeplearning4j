package org.deeplearning4j.cli.conf;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.deeplearning4j.cli.subcommands.Generate;
import org.junit.Test;

public class TestModelConfigurationUtil {

	public String getProp( Properties c, String key ) {
		
		String ret = "";
		
        if ( null != c.get( key )) {
        	
        	ret = (String) c.getProperty( key );
        	        	
        }
        
        return ret;
		
	}
	
	public Properties loadProps( String confFile ) {

		Properties configProps = new Properties();
        
//System.out.println( "loading conf file: " + this.configurationFile );
        InputStream in = null;
        try {
            in = new FileInputStream( confFile );
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        try {
            configProps.load(in);
            in.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }		

        return configProps;
        
	}
	
	@Test
	public void testDBNModelArchitectureGenerate() {
		
		String conf_file = "src/test/resources/generate/architectures/dbn/conf/dbn_generate_conf.txt";
		
		ModelConfigurationUtil util = new ModelConfigurationUtil( loadProps( conf_file ) );
		util.generateDefaultDBNConfigFile();
		
		
	}

	@Test
	public void testCNNModelArchitectureGenerate() {
		
		String conf_file = "src/test/resources/generate/architectures/cnn/conf/cnn_generate_conf.txt";
		
		ModelConfigurationUtil util = new ModelConfigurationUtil( loadProps( conf_file ) );
		util.generateDefaultCNNConfigFile();
		
		
	}
	
	@Test
	public void testSerdeMechanics() {
		
	//	ModelConfigurationUtil util = new ModelConfigurationUtil();
	//	util.generateDefaultDBNConfigFile("");
		
		
	}
	
	
	
}
