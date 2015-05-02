/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.ui.tsne;

import io.dropwizard.views.View;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.ui.nearestneighbors.NearestNeighborsQuery;
import org.deeplearning4j.ui.uploads.FileResource;
import org.glassfish.jersey.media.multipart.FormDataContentDisposition;
import org.glassfish.jersey.media.multipart.FormDataParam;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 10/8/14.
 */
@Path("/tsne")
@Produces(MediaType.TEXT_HTML)
public class TsneResource extends FileResource {
    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger(FileResource.class);
    private String path;
    /**
     * The file path for uploads
     *
     * @param filePath the file path for uploads
     */
    public TsneResource(String filePath) {
        super(filePath);
    }

    @GET
    public View get() {
        return new TsneView();
    }

    private List<String> coords;

    @POST
    @Path("/coords")
    public Response coords() {

        if(coords.isEmpty())
            throw new IllegalStateException("Unable to get coordinates; empty");

        return Response.ok(coords).build();
    }

    public void setPath(String path) throws IOException {
        coords = FileUtils.readLines(new File(path));
    }
    @Override
    public void handleUpload(File path) {
            try {
            setPath(path.getAbsolutePath());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

}
