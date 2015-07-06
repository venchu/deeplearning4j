package org.deeplearning4j.cli.driver;


import org.deeplearning4j.cli.subcommands.Generate;
import org.deeplearning4j.cli.subcommands.Train;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.spi.SubCommand;
import org.kohsuke.args4j.spi.SubCommandHandler;
import org.kohsuke.args4j.spi.SubCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CLI Driver for dl4j.
 *
 * Supports the train command
 *
 * @author Adam Gibson
 * @author Josh Patterson
 */
public class CommandLineInterfaceDriver {

	private static Logger log = LoggerFactory.getLogger(CommandLineInterfaceDriver.class);

	@Argument(required=true,metaVar="action",usage="subcommands, e.g., {generate|train}",handler=SubCommandHandler.class)
	@SubCommands({
			@SubCommand(name="train",impl=Train.class),
			@SubCommand(name="generate",impl=Generate.class)
	})
	private org.deeplearning4j.cli.subcommands.SubCommand subCommand;



    /**
     * Print the usage for the command.
     */
	public static void printUsage() {
/*        log.info( "Usage: " );
		log.info( "\tdl4j [command] [params] " );
		log.info( "Commands: " );
		log.info( "\ttrain\tbuild a deep learning model " );
		log.info( "\ttest\ttest a deep learning model " );
		log.info( "\tpredict\tscore new records against a deep learning model " );
		log.info( "" );
*/
		
    	System.out.println( "DL4J: Command-Line Deep Learning Modeling" );
    	System.out.println( "" );
    	System.out.println( "\tUsage:" );
    	System.out.println( "\t\tdl4j <command> <flags>" );
    	System.out.println( "" );
    	System.out.println( "\tCommands:" );
    	System.out.println( "\t\tgenerate\t\tGenerates a model configuration for a specific network architecture" );
    	System.out.println( "\t\ttrain\t\tGenerates a model configuration for a specific network architecture" );
    	System.out.println( "" );
    	System.out.println( "\tExample:" );
    	System.out.println( "\t\tdl4j generate -conf /tmp/iris_conf.txt \t\t (generates a model configuration file for the network architecture defined in iris_conf.txt)" );
    	System.out.println( "\t\tdl4j train -conf /tmp/iris_conf.txt " );
    	
		
		
	}

    public void doMain(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        try {
        	
            parser.parseArgument(args);
            subCommand.execute();
        } catch( CmdLineException e ) {
            System.err.println(e.getMessage());
        	printUsage();
            return;
        }
    }


    public static void main(String [] args) throws Exception {
        new CommandLineInterfaceDriver().doMain(args);

    }

}
