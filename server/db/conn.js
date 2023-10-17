const mongoose = require('mongoose');

const DB = process.env.DATABASE;

const { MongoClient, ServerApiVersion } = require('mongodb');
const uri = process.env.DATABASE;

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(uri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
    tls: true,
    useUnifiedTopology: true
  }
});

async function run() {
  try {
    // Connect the client to the s erver (optional starting in v4.7)
    // console.log('Before connect');
    // await client.connect();
    // console.log('After connect');
    // // Send a ping to confirm a successful connection
    // await client.db("admin").command({ ping: 1 });
    // console.log("Pinged your deployment. You successfully connected to MongoDB!");

    // Connect to Mongoose
    await mongoose.connect(DB);
    console.log(`Connection Successfull`);
  } finally {
    // Ensures that the client will close when you finish/error
    await client.close();
  }
}
run().catch(console.dir);