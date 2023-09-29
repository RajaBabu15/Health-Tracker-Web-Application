const express = require('express');
const app_project = express.Router();

app_project.post('/predict',(req,res)=>{
    res.status(200).json({req:req.body,msg:"Request Accepted"});
});

module.exports = app_project;