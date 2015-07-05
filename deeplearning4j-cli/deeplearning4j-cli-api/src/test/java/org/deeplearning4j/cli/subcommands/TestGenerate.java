package org.deeplearning4j.cli.subcommands;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestGenerate {

	@Test
	public void test() {

		String conf_file = "src/test/resources/architectures/dbn/conf/dbn_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Generate cmd = new Generate( args );
/*		try {
			cmd.loadConfigFile();
		} catch (Exception e) {
			System.out.println( "could not load conf: " + e );
		}
*/
		cmd.execute();
		
		//System.out.println("[testLoadInputFormat] End");
		
		
	}

}
