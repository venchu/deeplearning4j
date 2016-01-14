package org.deeplearning4j.models.sequencevectors;

import org.canova.api.berkeley.Pair;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW;
import org.deeplearning4j.models.sequencevectors.classes.Transaction;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.math.BigDecimal;
import java.util.*;

import static org.junit.Assert.*;

/**
 * This is going to be basic for SequenceVectors graph transformation
 *
 * @author raver119@gmail.com
 */
public class SequenceVectorsGraphTest {

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    /**
     * In this test/example we'll assume that we have csv file that contains transactions sorted by <source asc, time asc>
     * @throws Exception
     */
    @Test
    public void testFit() throws Exception {
        // we load data from csv file, line by line
        SentenceIterator lineIterator = new BasicLineIterator(new ClassPathResource("/transactions/train.csv").getFile());

        List<Sequence<Transaction>> sequences = new ArrayList<>();

        // skip header line
        lineIterator.nextSentence();

        // we roll over them, until we get new account ID
        Sequence<Transaction> currentSequence = new Sequence<Transaction>();
        List<String> labels = new ArrayList<>();
        long lastId = 0;
        while (lineIterator.hasNext()) {
            String line = lineIterator.nextSentence();

            String[] desc = line.split(";");

            if (lastId == 0) lastId = Long.valueOf(desc[0]);
            /*
                our csv schema is simple:
                0. account ID
                1. good type
                2. pos terminal id
                3. time in milliseconds
                4. transaction code 1
                5. transaction code 2
                6. volume, if applicable
                7. sum
             */
            Transaction transaction = null;
            try {
                 transaction = new Transaction(
                        Integer.valueOf(desc[2]), // POS terminal number
                        Long.valueOf(desc[3]),  // time for this transaction, in YYYYMMDDHHMMSS format
                        Integer.valueOf(desc[1]), // goods id
                        Integer.valueOf(desc[4]),  // transaction code 1
                        new BigDecimal(desc[6].replaceAll("\"", "")) // sum of the transaction
                );
            } catch (Exception e) {
                System.out.println("Skipping malformed line: " + line);
                continue;
            }

            transaction.setAccountId(Long.valueOf(desc[0]));

            currentSequence.addElement(transaction);

            long clientId = transaction.getAccountId();
            if (lastId != clientId) {
                if (!labels.contains(String.valueOf(lastId))) {
                    labels.add(String.valueOf(lastId));
                }
                System.out.println("Saving sequence for account: [" + lastId + "], transactions #: ["+ currentSequence.getElements().size()+"]");
                sequences.add(currentSequence);
                // at this moment we build graph for latest account

                // and transform this graph into a sequence of Transactions, based on angle between nodes in graph

                currentSequence.setSequenceLabel(new Transaction(lastId));
                currentSequence = new Sequence<Transaction>();
            }
            lastId = clientId;
        }



        AbstractSequenceIterator<Transaction> iterator = new AbstractSequenceIterator.Builder<Transaction>(sequences)
                .build();


        SequenceVectors<Transaction> vec = new SequenceVectors.Builder<Transaction>()
                .useAdaGrad(false)
                .batchSize(100)
                .layerSize(150)
                .learningRate(0.025)
                .iterate(iterator)
                .negativeSample(5)
                .iterations(10)
                .epochs(3)
                .elementsLearningAlgorithm(new SkipGram<Transaction>())
                .sequenceLearningAlgorithm(new DBOW<Transaction>())
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .build();

        vec.fit();


        /*
            At this moment we'll assume our model is built, and we'll start testing it.

            First test will be synthetic, we'll pass in two transactions, and check their similarity
         */

        Transaction transaction1 = new Transaction(95, 20150506203000L,7, 3, new BigDecimal(100));
        Transaction transaction2 = new Transaction(24, 20150505203000L,1, 3, new BigDecimal(100));

        Transaction transaction3 = new Transaction(24, 20150505203000L,7, 3, new BigDecimal(100));

        Transaction transaction4 = new Transaction(24, 20150505103000L,7, 3, new BigDecimal(100));

        double sim1 = vec.similarity(transaction1.getLabel(), transaction2.getLabel());
        System.out.println("Similarity 1: " + sim1);

        double sim2 = vec.similarity(transaction2.getLabel(), transaction3.getLabel());
        System.out.println("Similarity 2: " + sim2);

        double sim3 = vec.similarity(transaction3.getLabel(), transaction4.getLabel());
        System.out.println("Similarity 3: " + sim3);

        double simA1 = vec.similarity("555500088992" , "555500088998");
        System.out.println("Similarity between accounts 1: " + simA1);

        double simA2 = vec.similarity("555500088992" , "555500090784");
        System.out.println("Similarity between accounts 2: " + simA2);

        double simA3 = vec.similarity("555500088998" , "555500090784");
        System.out.println("Similarity between accounts 3: " + simA3);

        System.out.println("-----------------------------");

        double simB1 = vec.similarity("555500032180" , "555500032477");
        System.out.println("Similarity between bad accounts 1: " + simB1);

        double simB2 = vec.similarity("555500040462" , "555500065777");
        System.out.println("Similarity between bad accounts 2: " + simB2);

        Collection<Pair<String, Double>> list = nearestAccountsFor(vec, "555500032477", labels, 10); //vec.wordsNearest(new ArrayList<String>(Arrays.asList(new String[]{"555500032477"})), new ArrayList<String>(), 10 );
        System.out.println("Nearest for [555500032477]: ");
        for (Pair<String, Double> line: list) {
            System.out.println("               ["+line.getFirst()+"] > "+line.getSecond()+" ");
        }
    }

    private Collection<Pair<String, Double>> nearestAccountsFor(SequenceVectors vec, String account, List<String> accounts, int number) {
        List<Pair<String, Double>> results = new ArrayList<>();
        for (String label: accounts) {
            double sim = vec.similarity(account, label);
            results.add(Pair.makePair(account, sim));
        }

        results.sort(new SimComp());

        return results.subList(0, 10);
    }

    private class SimComp implements Comparator<Pair<String, Double>> {

        @Override
        public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
            return Double.compare(o1.getSecond(), o2.getSecond());
        }
    }
}