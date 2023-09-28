const dotenv= require("dotenv");
const mongoose = require('mongoose');
const express = require('express');
const app = express();


dotenv.config({path:'./config.env'});
require('./db/conn');
// const User = require('./model/userSchema');

app.use(express.json());

app.use(require('./router/auth'));

const PORT = process.env.PORT;


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

app.listen(PORT, () => {
    console.log(`Server is Running at the port ${PORT}`);
});
