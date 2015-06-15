package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class ParamInitializerTests {

    @Test
    public void testDefault() {
        ParamInitializer initializer = new DefaultParamInitializer();
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().layer(new RBM())
                .nIn(4).nOut(4).build();
        Map<String,INDArray> test = new HashMap<>();
        initializer.init(test, conf);
        assertTrue(Arrays.equals(test.get(DefaultParamInitializer.WEIGHT_KEY).shape(), new int[]{4,4}));
    }

}
