const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    minlength: 3,
    trim: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    
  },
  phone: {
    type: Number,
    required: true,
    minlength: 10,
    trim: true,
  },
  work: {
    type: String,
    required: true,
    trim: true,
  },
  password: {
    type: String,
    required: true,
    minlength: 8,
    trim: true,
  },
  cpassword: {
    type: String,
    required: true,
    minlength: 8,
    trim: true,
  },
  tokens:[
    {
        token:{
            type: String,
            required: true,
        }
    }
  ]
});

  

userSchema.pre('save', async function (next) {
  
    // Hash the password and the cpassword.
    this.password = await bcrypt.hash(this.password, 12);
    this.cpassword = await bcrypt.hash(this.cpassword, 12);
  
    // Continue with the save.
    next();
  });
  

userSchema.methods.generateAuthToken = async function(){
    try{
        let token = jwt.sign({_id:this._id},process.env.SECRET_KEY);
        this.tokens=this.tokens.concat({token:token});
        await this.save();
        return token;
    }catch(err){
        console.log(err);
    }
}


const User = mongoose.model('USER', userSchema);

module.exports = User;
