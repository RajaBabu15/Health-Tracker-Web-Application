const express = require("express");
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const router = express.Router();

require("../db/conn");
const User = require("../model/userSchema");

router.get("/", (req, res) => {
  res.send(`Hello world from the server RouterJs`);
});

router.post("/register", async (req, res) => {
  const { name, email, phone, work, password, cpassword } = req.body;
  if (!name || !email || !phone || !work || !password || !cpassword) {
    return res.status(422).json({ error: "Please Fill the field properly" });
  }

  try {   
    const user = new User({ name, email, phone, work, password, cpassword });
    const userExist = await User.findOne({ email: email });
    if (userExist) {
      return res.status(422).json({ error: "Email already Exist." });
    } else if (password != cpassword) {
      return res
        .status(422)
        .json({
          error: "Password Field and Confirm Password Field must be same.",
        });
    }
    
    await user.save();

    res.status(201).json({ message: "User Registered Successfully" });
  } catch (err) {
    console.log(err);
    res.status(422).json(err);
  }
});

router.post("/signin", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      res.status(422).json({ message: "Please enter the credential" });
    }

    const userLogin = await User.findOne({ email: email });

    const isMatch = await bcrypt.compare(password,userLogin.password);
    console.log({pss:userLogin.password,isMatch,password});

    if (!userLogin | !isMatch)
      return res.status(400).json({ message: "Invalid Credentials" });

    const token = await userLogin.generateAuthToken();
    res.cookie("jwtoken",token,{
      expires:new Date(Date.now()+2592000000),
      httpOnly:true
    });

    res.json({ message: "User SignIn SuccessFully", userDocument: userLogin });
  } catch (err) {
    console.dir(err);
  }
});

module.exports = router;
