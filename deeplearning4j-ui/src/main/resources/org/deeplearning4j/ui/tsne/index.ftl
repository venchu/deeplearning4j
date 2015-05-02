<!doctype html>
<html lang="en" data-framework="react">
	<head>
		<meta charset="utf-8">
		<title>TSNE</title>
        <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap-theme.min.css">
		<#--<link rel="stylesheet" href="node_modules/todomvc-common/base.css">-->
		<#--<link rel="stylesheet" href="node_modules/todomvc-app-css/index.css">-->

        <!-- jQuery -->
        <script src="https://code.jquery.com/jquery-2.1.3.min.js"></script>
        <script src="/assets/jquery-fileupload.js"></script>
        <script src="/assets/bootstrap-3.3.4-dist/js/bootstrap.min.js"></script>
        <script src="/assets/d3.min.js"></script>
        <script src="/assets/js/tsne/app.js"></script>
        <script src="/assets/render.js"></script>
        <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>




        <style>
            body {
                font-family: 'Roboto', sans-serif;
                color: #333;
                font-weight: 300;
                font-size: 16px;
            }
            svg {
                border: 1px solid #333;
            }
            #wrap {
                width: 800px;
                margin-left: auto;
                margin-right: auto;
            }
            #embed {
                margin-top: 10px;
            }
            h1 {
                text-align: center;
                font-weight: normal;
            }
            .tt {
                margin-top: 10px;
                background-color: #EEE;
                border-bottom: 1px solid #333;
                padding: 5px;
            }
            .txth {
                color: #F55;
            }
            .cit {
                font-family: courier;
                padding-left: 20px;
                font-size: 14px;
            }
        </style>

        <script>


        </script>

	</head>
	<body>
    <div id="container">
        <div id="wrapper">
            <div id="page-content-wrapper">
                <div class="container-fluid">
                    <h1 style="text-align: center; font-size: 400%">Deeplearning4j</h1>
                    <hr>
                    <h2>t-Distributed Stochastic Neighbor Embedding (t-SNE)</h2>
                    <h4>
                        <p id="tsne-description">Upload a text file. The site will render a t-Distributed Stochastic Neighbor visualization (aka word cloud).</p>
                        <p id="tsne-loaded" hidden>Below is the t-Distributed Stochastic Neighbor visualization.</p>
                    </h4>
                    <br>
                    <div class="row" id="upload">
                        <form encType="multipart/form-data" action="/tsne/upload" method="POST" id="form">
                            <input name="file" type="file">
                            <br>
                            <input type="submit">
                        </form>
                    </div>
                </div>
            </div>

        </div>
    </div>
	</body>
</html>
