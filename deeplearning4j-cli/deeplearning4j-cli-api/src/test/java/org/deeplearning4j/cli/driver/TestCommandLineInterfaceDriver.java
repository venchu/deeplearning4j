package org.deeplearning4j.cli.driver;

import static org.junit.Assert.*;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

public class TestCommandLineInterfaceDriver {

	@Test
	public void testMainCLIDriverEntryPoint_NoArgs() throws Exception {

		String[] args = {  };

		CommandLineInterfaceDriver.main( args );

	}
	
	@Test
	public void testMainCLIDriverEntryPoint_Train_NoArgs() throws Exception {

		String[] args = { "train" };

		CommandLineInterfaceDriver.main( args );

	}
	
	
	@Test
	public void testMainCLIDriverEntryPoint_Train() throws Exception {

		String[] args = { "train", "-conf", "src/test/resources/train/architectures/dbn/conf/dbn_test_conf.txt" };

		CommandLineInterfaceDriver.main( args );

	}

	@Test
	public void testMainCLIDriverEntryPoint_Generate_NoArgs() throws Exception {

		String[] args = { "generate" };

		CommandLineInterfaceDriver.main( args );

	}
	
	@Test
	public void testMainCLIDriverEntryPoint_Generate() throws Exception {

		String[] args = { "generate", "-conf", "src/test/resources/generate/architectures/dbn/conf/dbn_generate_conf.txt" };

		CommandLineInterfaceDriver.main( args );
/*
		String outputFile = "csv/data/uci_iris_sample.txt";

		ArrayList<String> vectors = new ArrayList<>();

		Map<String, Integer> labels = new HashMap<>();
		List<String> lines = FileUtils.readLines(new ClassPathResource(outputFile).getFile());
		for(String line : lines) {
			// process the line.
			if (!line.trim().isEmpty()) {
				vectors.add( line );

				String parts[] = line.split(" ");
				String key = parts[0];
				if (labels.containsKey(key)) {
					Integer count = labels.get(key);
					count++;
					labels.put(key, count);
				} else {
					labels.put(key, 1);
				}

			}
		}

		assertEquals(12, vectors.size());
		assertEquals(12, labels.size());
        File f = new File("/tmp/iris_unit_test_sample.txt");
        f.deleteOnExit();
        assertTrue(f.exists());
*/

	}
}
