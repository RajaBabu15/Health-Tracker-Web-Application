const express = require('express');
const app = express();

const middleware = (req,res,next)=>{
    console.log(`Hello my middleware`);
    next();
}


app.get('/', (req, res) => {
    res.send(`Hello world from the server`);
});

app.get('/about', (req, res) => {
    res.send(`Hello About world from the server`);
});

app.get('/contact', (req, res) => {
    res.send(`Hello Contact world from the server`);
});

app.get('/signin',middleware, (req, res) => {
    res.send(`Hello SignIn world from the server`);
});

app.get('/signup', (req, res) => {
    res.send(`Hello Registration world from the server`);
});

app.listen(3000, () => {
    console.log(`Server is Running at the port 3000`);
});
