const dotenv= require("dotenv");
const mongoose = require('mongoose');
const express = require('express');
const app = express();


dotenv.config({path:'./config.env'});

require('./db/conn'); // connecting to the mongo db
const project=require('./router/api/project'); // calling to the predict function 
app.use(require('./router/auth'));

app.use(express.json());

app.use('/',project);


const PORT = process.env.PORT || 5000;


const middleware = (req,res,next)=>{
    console.log(`Hello my middleware`);
    next();
}




app.get('/', (req, res) => {
    res.send(`Hello world from the server app.js`);
});

app.get('/about',middleware, (req, res) => {
    res.send(`Hello About world from the server`);
});

app.get('/contact', (req, res) => {
    res.send(`Hello Contact world from the server`);
});

app.get('/signin', (req, res) => {
    res.send(`Hello SignIn world from the server`);
});

app.get('/signup', (req, res) => {
    res.send(`Hello Registration world from the server`);
});


if(process.env.NODE_ENV == "production"){
    app.use(express.static('client/build'));
}

app.listen(PORT, () => {
    console.log(`Server is Running at the port ${PORT}`);
});
