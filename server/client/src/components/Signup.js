import React, { useState, useEffect, useRef } from "react";
import { NavLink,useNavigate } from "react-router-dom";


const Signup = () => {
    const navigate = useNavigate();
  const [name, setName] = useState("");
  const nameRef = useRef();

  const [email, setEmail] = useState("");
  const emailRef = useRef();

  const [phone, setPhone] = useState("");
  const phoneRef = useRef();

  const [work, setWork] = useState("");
  const workRef = useRef();

  const [password, setPassword] = useState("");
  const passwordRef = useRef();

  const [cpassword, setcpassword] = useState("");
  const cpasswordRef = useRef();

  useEffect(() => {
    const nameElement = nameRef.current;
    if (nameElement && nameElement.focus) {
      if (name === "") {
        nameElement.style.notch = "true";
      } else {
        nameElement.style.notch = "false";
      }
    }

    const emailElement = emailRef.current;
    if (emailElement && emailElement.focus) {
      if (email === "") {
        emailElement.style.notch = "true";
      } else {
        emailElement.style.notch = "false";
      }
    }

    const phoneElement = phoneRef.current;
    if (phoneElement && phoneElement.focus) {
      if (phone === "") {
        phoneElement.style.notch = "true";
      } else {
        phoneElement.style.notch = "false";
      }
    }

    const workElement = workRef.current;
    if (workElement && workElement.focus) {
      if (work === "") {
        workElement.style.notch = "true";
      } else {
        workElement.style.notch = "false";
      }
    }

    const passwordElement = passwordRef.current;
    if (passwordElement && passwordElement.focus) {
      if (password === "") {
        passwordElement.style.notch = "true";
      } else {
        passwordElement.style.notch = "false";
      }
    }

    const cpasswordElement = cpasswordRef.current;
    if (cpasswordElement && cpasswordElement.focus) {
      if (cpassword === "") {
        cpasswordElement.style.notch = "true";
      } else {
        cpasswordElement.style.notch = "false";
      }
    }
  }, [name, email, phone, work, password, cpassword]);

  const PostData = async (e) =>{
    console.log(e);
    console.log("Fucker it");
    const res = await fetch('/register',{
        method: "POST",
        headers:{
             "Content-Type":"application/json"
        },
        body:JSON.stringify({
            name, email, phone, work, password, cpassword
        })
    });
    
    const data = await res.json();
    if(data.status === 422 || !data) {
        window.alert("Invalid Registration");
        console.log("Invalid Registration");
    }else {
        window.alert("valid Registration");
        console.log("valid Registration");
        navigate('/login');
    }
  };
  return (
    <>
      <section className="vh-100" styles="background-color: #eee;">
        <div className="container h-100">
          <div className="row d-flex justify-content-center align-items-center h-100">
            <div className="col-lg-12 col-xl-11">
              <div className="card text-black" styles="border-radius: 25px;">
                <div className="card-body p-md-5">
                  <div className="row justify-content-center">
                    <div className="col-md-10 col-lg-6 col-xl-5 order-2 order-lg-1">
                      <p className="text-center h1 fw-bold mb-5 mx-1 mx-md-4 mt-4">
                        Sign up
                      </p>

                      <form method="POST" className="mx-1 mx-md-4">
                        <div className="d-flex flex-row align-items-center mb-4">
                          <i className="fas fa-user fa-lg me-3 fa-fw"></i>
                          <div className="form-outline flex-fill mb-0">
                            <input
                              type="text"
                              id="name"
                              name="name"
                              className="form-control"
                              ref={nameRef}
                              value={name}
                              onChange={(e) => setName(e.target.value)}
                            />
                            {name === "" && (
                              <label
                                className="form-label"
                                for="name"
                                autoCorrect="off"
                              >
                                Your Name
                              </label>
                            )}
                          </div>
                        </div>
                        <div className="d-flex flex-row align-items-center mb-4">
                          <i className="fas fa-envelope fa-lg me-3 fa-fw"></i>
                          <div className="form-outline flex-fill mb-0">
                            <input
                              type="email"
                              id="email"
                              name="email"
                              className="form-control"
                              ref={emailRef}
                              value={email}
                              onChange={(e) => setEmail(e.target.value)}
                            />
                            {email === "" && (
                              <label
                                className="form-label"
                                for="email"
                              >
                                Your Email
                              </label>
                            )}
                          </div>
                        </div>

                        <div className="d-flex flex-row align-items-center mb-4">
                          <i className="fas fa-phone fa-lg me-3 fa-fw"></i>
                          <div className="form-outline flex-fill mb-0">
                            <input
                              type="tel"
                              id="phone"
                              name="phone"
                              className="form-control"
                              ref={phoneRef}
                              value={phone}
                              onChange={(e) => setPhone(e.target.value)}
                            />
                            {phone === "" && (
                              <label
                                className="form-label"
                                for="phone"
                              >
                                Phone
                              </label>
                            )}
                          </div>
                        </div>

                        <div className="d-flex flex-row align-items-center mb-4">
                          <i className="fas fa-briefcase fa-lg me-3 fa-fw"></i>
                          <div className="form-outline flex-fill mb-0">
                            <input
                              type="text"
                              id="work"
                              name="work"
                              className="form-control"
                              ref={workRef}
                              value={work}
                              onChange={(e) => setWork(e.target.value)}
                            />
                            {work === "" && (
                              <label
                                className="form-label"
                                for="work"
                              >
                                Work
                              </label>
                            )}
                          </div>
                        </div>

                        <div className="d-flex flex-row align-items-center mb-4">
                          <i className="fas fa-lock fa-lg me-3 fa-fw"></i>
                          <div className="form-outline flex-fill mb-0">
                            <input
                              type="password"
                              id="password"
                              name="password"
                              className="form-control"
                              ref={passwordRef}
                              value={password}
                              onChange={(e) => setPassword(e.target.value)}
                            />
                            {password === "" && (
                              <label
                                className="form-label"
                                for="password"
                              >
                                Password
                              </label>
                            )}
                          </div>
                        </div>

                        <div className="d-flex flex-row align-items-center mb-4">
                          <i className="fas fa-key fa-lg me-3 fa-fw"></i>
                          <div className="form-outline flex-fill mb-0">
                            <input
                              type="password"
                              id="cpassword"
                              name="cpassword"
                              className="form-control"
                              ref={cpasswordRef}
                              value={cpassword}
                              onChange={(e) =>
                                setcpassword(e.target.value)
                              }
                            />
                            {cpassword === "" && (
                              <label
                                className="form-label"
                                for="cpassword"
                              >
                                Repeat your password
                              </label>
                            )}
                          </div>
                        </div>

                        <div className="form-check d-flex justify-content-center mb-5">
                          <input
                            className="form-check-input me-2"
                            type="checkbox"
                            value=""
                            id="form2Example3c"
                          />
                          <label
                            className="form-check-label"
                            for="form2Example3"
                          >
                            I agree all statements in{" "}
                            <a href="#!">Terms of service</a>
                          </label>
                        </div>

                        <div className="d-flex justify-content-center mx-4 mb-3 mb-lg-4">
                          <button
                            type="button"
                            name="signup"
                            id="signup"
                            onClick={PostData}
                            className="btn btn-primary btn-lg form-submit"
                          >
                            Register
                          </button>
                        </div>
                      </form>
                      <div className="d-flex justify-content-center mx-4 mb-3 mb-lg-4">
                        <NavLink to="/login" className="text-decoration-none">
                          Already have an account? Login here.
                        </NavLink>
                      </div>
                    </div>
                    <div className="col-md-10 col-lg-6 col-xl-7 d-flex align-items-center order-1 order-lg-2">
                      <img
                        src="https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-registration/draw1.webp"
                        className="img-fluid"
                        alt="Sample image"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
};

export default Signup;
