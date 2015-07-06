package org.deeplearning4j.cli.subcommands;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.junit.Test;
//import org.nd4j.linalg.dataset.api.DataSet;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

public class TestTrainCommand {

	/**
	 * Make sure we're loading the training process configuration file
	 * 
	 */
	@Test
	public void testLoadConf() {


		String conf_file = "src/test/resources/train/architectures/dbn/conf/dbn_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Train cmd = new Train( args );
		//cmd.execute();
		try {
			cmd.loadConfigFile();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		cmd.debugPrintConf();
		
		// dl4j.input.format
		String inputFormatClass = cmd.configProps.getProperty(Train.INPUT_FORMAT_KEY, "");
		assertEquals( "org.canova.api.formats.input.impl.SVMLightInputFormat", inputFormatClass );
		
		// dl4j.input.format
		String architectureJsonFile = cmd.configProps.getProperty("dl4j.model.config.architecture", "");
		assertEquals( "dbn", architectureJsonFile );
		
		
	}

	/**
	 * Make sure we're loading the network architecture json file
	 * 
	 */
	@Test
	public void testLoadModelArchitecture() {


		String conf_file = "src/test/resources/train/architectures/dbn/conf/dbn_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Train cmd = new Train( args );
		//cmd.execute();
		try {
			cmd.loadConfigFile();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		
		cmd.debugPrintConf();
		cmd.validateModelConfigFile();
		
		assertEquals( true, cmd.validModelConfigJSONFile );
	
	}
	
	/**
	 * Test loading the conf, network arch, and then train on a dataset
	 * 
	 */
	@Test
	public void testFullTrainProcessDBNIris() {

		String conf_file = "src/test/resources/train/architectures/dbn/conf/dbn_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Train cmd = new Train( args );
		cmd.execute();
	
	}
	
	/**
	 * Double check some mechanics of svmlight readers
	 * 
	 * @throws IOException
	 */
	@Test
	public void testRR() throws IOException {
		
		String conf_file = "src/test/resources/train/architectures/dbn/conf/dbn_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Train cmd = new Train( args );

        File inputFile = new File( "src/test/resources/data/irisSvmLight.txt" );
        InputSplit split = new FileSplit( inputFile );
        InputFormat inputFormat = cmd.createInputFormat();

        RecordReader reader = null;

        try {
            reader = inputFormat.createReader(split);
        } catch (Exception e) {
            e.printStackTrace();
        }
		
		
        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File( "src/test/resources/train/architectures/dbn/conf/model_arch.json" )));
        DataSetIterator iter = new RecordReaderDataSetIterator( reader , conf.getConf(0).getBatchSize(),-1,conf.getConf(conf.getConfs().size() - 1).getNOut());
		
        assertEquals(true, iter.hasNext());
        	
        	DataSet d = iter.next();
        	
        //	System.out.println( "cols: " + d.getFeatures().columns() );
        	//System.out.println( "cols: " + d.getFeatures().rows() );
        	
        	assertEquals( 4, d.getFeatures().columns() );
        	assertEquals( 12, d.getFeatures().rows() );
        
        
		
	}
	
}
