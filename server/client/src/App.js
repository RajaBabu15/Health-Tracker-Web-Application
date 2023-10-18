import React, { useState, useEffect } from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import "bootstrap/dist/css/bootstrap.css";
import Contact from "./components/Contact";
import About from "./components/About";
import Login from "./components/Login";
import Signup from "./components/Signup";
import ErrorPage from "./components/ErrorPage";
import Predictor from "./components/predictor";
import { BrowserRouter as  Routes, Route, Link, useNavigate } from "react-router-dom";

const App = () => {
  const navigate = useNavigate();
  const [state, setState] = useState(/* initial state */);

  // Reset state when component unmounts
  useEffect(() => {
    return () => {
      setState(/* reset state */);
    };
  }, []);

  const handleClick = () => {
    // Navigate to Predictor component
    navigate('/predict');
  };

  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/about" element={<About />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="*" element={<ErrorPage />} />
        <Route path="/predict" element={<Predictor />} />
      </Routes>
    </>
  );
};

export default App;
