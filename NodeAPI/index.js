const express = require('express')
const bodyParser = require('body-parser');
const request = require('request');

const app = express()
const port = 80;

// Get the mongodb url
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb+srv://dbuser:nlp2021@clusternlp.5iof9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority";

app.set('view engine', 'ejs')

app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());


app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})


app.get('/', (req, res) => {
    res.render('index', {predictions:null})
})

app.post('/predict', (req,res) => {
    request.post({url:'http://127.0.0.1:5000/predict',formData:{
        text:req.body.plot
    }}, function (error, response, body) {
        console.error('error:', error); // Print the error
        console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
        console.log('body:', body); // Print the data received
        console.log(JSON.parse(body))
        console.log(JSON.parse(body).predictionText)
        // res.send(JSON.parse(JSON.parse(body).prediction)); //Display the response on the website

        // Connect to the db
        MongoClient.connect(url, function(err, db) {
          if(err) { return console.dir(err); }

          // Create collection if does not exist
          var collection = db.db('ClusterNLP').collection('Pred_Records');

          // Get the data to be stored
          var plot = req.body.plot;
          var prediction = JSON.parse(JSON.parse(body).predictionText);

          // Get rcord to be inserted
          var rec = {'Plot':plot, 'Prediction': prediction};
          
          // Insert the record to the collection, throw error if insert unsuccessful
          collection.insert(rec, {w:1}, function(err, result) {});
        });

        res.render('predictions',{predictions:JSON.parse(JSON.parse(body).predictionText)})

        
      });      
    console.log(req.body.plot)
    // res.render('index', {});

});

app.get('/submissions', function (req, res) {
  var resultArray = [];
  MongoClient.connect(url, function(err, db) {
    if(err) { return console.dir(err); }

    var cursor = db.db('ClusterNLP').collection('Pred_Records').find();
    cursor.forEach(function(doc, err){
      resultArray.push(doc)
    }, function(){
      res.render('submissions', {items: resultArray});
    });
  });
  
});