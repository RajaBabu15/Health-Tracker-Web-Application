import {
  MDBInput,
  MDBCheckbox,
  MDBBtn,
  MDBValidation,
  MDBValidationItem,
  MDBTextArea,
} from "mdb-react-ui-kit";
import React from "react";

const Contact = () => {
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
                      
                      <MDBValidation
                        noValidate
                        id="form"
                        className="text-center"
                        style={{ width: "100%", maxWidth: "300px" }}
                      >
                        <h2>Contact us</h2>

                        <MDBValidationItem
                          invalid
                          feedback="Please provide your name."
                        >
                          <MDBInput
                            label="Name"
                            v-model="name"
                            wrapperClass="mb-4"
                            required
                          />
                        </MDBValidationItem>

                        <MDBValidationItem
                          invalid
                          feedback="Please provide your email."
                        >
                          <MDBInput
                            type="email"
                            label="Email address"
                            v-model="email"
                            wrapperClass="mb-4"
                            required
                          />
                        </MDBValidationItem>

                        <MDBValidationItem
                          invalid
                          feedback="Please provide mail subject."
                        >
                          <MDBInput
                            label="Subject"
                            v-model="subject"
                            wrapperClass="mb-4"
                            required
                          />
                        </MDBValidationItem>

                        <MDBValidationItem
                          invalid
                          feedback="Please provide a message text."
                        >
                          <MDBTextArea
                            wrapperClass="mb-4"
                            label="Message"
                            required
                          />
                        </MDBValidationItem>

                        <MDBValidationItem feedback="">
                          <MDBCheckbox
                            wrapperClass="d-flex justify-content-center"
                            label="Send me copy"
                          />
                        </MDBValidationItem>

                        <MDBBtn
                          type="submit"
                          color="primary"
                          block
                          className="my-4"
                        >
                          Send
                        </MDBBtn>
                      </MDBValidation>
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
export default Contact;
