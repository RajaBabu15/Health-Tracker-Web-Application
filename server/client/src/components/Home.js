import React from "react";
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div>
      <h1>Hello World From Home</h1>
      <Link to="/predict">
        <button>Predict Mental Health</button>
      </Link>
    </div>
  );
};

export default Home;
