const express = require("express");
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
    const userExist = await User.findOne({ email: email });
    if (userExist) {
      return res.status(422).json({ error: "Email already Exist." });
    }
    const user = new User({ name, email, phone, work, password, cpassword });

    await user.save();

    res.status(201).json({ message: "User Registered Successfully" });
  } catch (err) {
    console.log(err);
  }
});

module.exports = router;
