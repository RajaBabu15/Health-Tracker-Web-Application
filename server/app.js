
const express = require('express');
const app = express();
const dotenv= require("dotenv");

dotenv.config({path:'./config.env'});

const DB = process.env.DATABASE;
const PORT = process.env.PORT;





const middleware = (req,res,next)=>{
    console.log(`Hello my middleware`);
    next();
}


app.get('/', (req, res) => {
    res.send(`Hello world from the server`);
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
