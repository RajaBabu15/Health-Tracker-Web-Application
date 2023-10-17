import React, { useState } from "react";
import { Form, FormGroup, Button } from "react-bootstrap";
import axios from "axios";

function Predictor() {
  const [inputs, setFormState] = useState({
    age: 15,
    gender: "Male",
    country: "United States",
    state: "",
    self_employed: "Yes",
    family_history: "Yes",
    work_interfere: "Often",
    no_employees: "1-5",
    remote_work: "Yes",
    tech_company: "Yes",
    benefits: "Yes",
    care_options: "Yes",
    wellness_program: "Yes",
    seek_help: "Yes",
    anonymity: "Yes",
    leave: "Very easy",
    mental_health_consequence: "Yes",
    phys_health_consequence: "Yes",
    coworkers: "Yes",
    supervisor: "Yes",
    mental_health_interview: "Yes",
    phys_health_interview: "Yes",
    mental_vs_physical: "Yes",
    obs_consequence: "Yes",
  });

  const handleChange = (event) => {
    setFormState({
      ...inputs,
      [event.target.name]: event.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log(inputs);
    try {
      const response = await axios.post(
        "http://localhost:5000/predict",
        inputs
      );
      const prediction = response.data.prediction;
      console.log(prediction);

      // Do something with the prediction
    } catch (error) {
      // Handle the error
    }
  };

  return (
    <div>
      <Form onSubmit={handleSubmit}>
        <FormGroup>
          <Form.Label htmlFor="age">Age</Form.Label>
          <Form.Control
            type="number"
            name="age"
            value={inputs.age}
            onChange={handleChange}
          />
        </FormGroup>

        <FormGroup>
          <Form.Label>Gender</Form.Label>
          <Form.Check
            type="radio"
            label="Male"
            name="gender"
            value="Male"
            checked={inputs.gender === "Male"}
            onChange={handleChange}
          />
          <Form.Check
            type="radio"
            label="Female"
            name="gender"
            value="Female"
            checked={inputs.gender === "Female"}
            onChange={handleChange}
          />
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="country">Country</Form.Label>
          <Form.Control
            as="select"
            name="country"
            value={inputs.country}
            onChange={handleChange}
          >
            <option value="United States">United States</option>
            <option value="Canada">Canada</option>
            <option value="United Kingdom">United Kingdom</option>
            <option value="Bulgaria">Bulgaria</option>
            <option value="France">France</option>
            <option value="Portugal">Portugal</option>
            <option value="Netherlands">Netherlands</option>
            <option value="Switzerland">Switzerland</option>
            <option value="Poland">Poland</option>
            <option value="Australia">Australia</option>
            <option value="Germany">Germany</option>
            <option value="Russia">Russia</option>
            <option value="Mexico">Mexico</option>
            <option value="Brazil">Brazil</option>
            <option value="Slovenia">Slovenia</option>
            <option value="Costa Rica">Costa Rica</option>
            <option value="Austria">Austria</option>
            <option value="Ireland">Ireland</option>
            <option value="India">India</option>
            <option value="South Africa">South Africa</option>
            <option value="Italy">Italy</option>
            <option value="Sweden">Sweden</option>
            <option value="Colombia">Colombia</option>
            <option value="Latvia">Latvia</option>
            <option value="Romania">Romania</option>
            <option value="Belgium">Belgium</option>
            <option value="New Zealand">New Zealand</option>
            <option value="Zimbabwe">Zimbabwe</option>
            <option value="Spain">Spain</option>
            <option value="Finland">Finland</option>
            <option value="Uruguay">Uruguay</option>
            <option value="Israel">Israel</option>
            <option value="Bosnia and Herzegovina">
              Bosnia and Herzegovina
            </option>
            <option value="Hungary">Hungary</option>
            <option value="Singapore">Singapore</option>
            <option value="Japan">Japan</option>
            <option value="Nigeria">Nigeria</option>
            <option value="Croatia">Croatia</option>
            <option value="Norway">Norway </option>
            <option value="Thailand">Thailand</option>
            <option value="Denmark"></option>
            <option value="Bahamas, The">Bahamas, The </option>
            <option value="Greece">Greece </option>
            <option value="Moldova">Moldova </option>
            <option value="Georgia">Georgia </option>
            <option value="China">China </option>
            <option value="Czech Republic">Czech Republic </option>
            <option value="Philippines">Philippines </option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label>State</Form.Label>
          <Form.Control
            type="text"
            name="state"
            value={inputs.state}
            onChange={handleChange}
          />
        </FormGroup>

        <FormGroup>
          <Form.Label>Self Employed</Form.Label>
          <Form.Check
            type="radio"
            label="Yes"
            name="self_employed"
            value="Yes"
            checked={inputs.self_employed === "Yes"}
            onChange={handleChange}
          />
          <Form.Check
            type="radio"
            label="No"
            name="self_employed"
            value="No"
            checked={inputs.self_employed === "No"}
            onChange={handleChange}
          />
        </FormGroup>

        <FormGroup>
          <Form.Label>Family History</Form.Label>
          <Form.Check
            type="radio"
            name="family_history"
            value="Yes"
            label="Yes"
            checked={inputs.family_history === "Yes"}
            onChange={handleChange}
          />
          <Form.Check
            type="radio"
            name="family_history"
            value="No"
            label="No"
            checked={inputs.family_history === "No"}
            onChange={handleChange}
          />
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="work_interfere">Work Interfere</Form.Label>
          <Form.Control
            as="select"
            name="work_interfere"
            value={inputs.work_interfere}
            onChange={handleChange}
          >
            <option value="Often">Often</option>
            <option value="Rarely">Rarely</option>
            <option value="Never">Never</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="no_employees">Number of Employees</Form.Label>
          <Form.Control
            as="select"
            name="no_employees"
            value={inputs.no_employees}
            onChange={handleChange}
          >
            <option value="1-5">1-5</option>
            <option value="6-25">6-25</option>
            <option value="26-100">26-100</option>
            <option value="100-500">100-500</option>
            <option value="500-1000">500-1000</option>
            <option value="More than 1000">More than 1000</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label>Remote Work</Form.Label>
          <Form.Check
            type="radio"
            label="Yes"
            name="remote_work"
            value="Yes"
            checked={inputs.remote_work === "Yes"}
            onChange={handleChange}
          />
          <Form.Check
            type="radio"
            label="No"
            name="remote_work"
            value="No"
            checked={inputs.remote_work === "No"}
            onChange={handleChange}
          />
        </FormGroup>

        <FormGroup>
          <Form.Label>Tech Company</Form.Label>
          <Form.Check
            type="radio"
            label="Yes"
            name="tech_company"
            value="Yes"
            checked={inputs.tech_company === "Yes"}
            onChange={handleChange}
          />
          <Form.Check
            type="radio"
            label="No"
            name="tech_company"
            value="No"
            checked={inputs.tech_company === "No"}
            onChange={handleChange}
          />
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="benefits">Benefits</Form.Label>
          <Form.Control
            as="select"
            name="benefits"
            value={inputs.benefits}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="Don't know">Don't know</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="care_options">Care Options</Form.Label>
          <Form.Control
            as="select"
            name="care_options"
            value={inputs.care_options}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="Not sure">Not sure</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="wellness_program">Wellness Program</Form.Label>
          <Form.Control
            as="select"
            name="wellness_program"
            value={inputs.wellness_program}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="Don't know">Don't know</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="seek_help">Seek Help</Form.Label>
          <Form.Control
            as="select"
            name="seek_help"
            value={inputs.seek_help}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="Don't know">Don't know</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="anonymity">Anonymity</Form.Label>
          <Form.Control
            as="select"
            name="anonymity"
            value={inputs.anonymity}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="Don't know">Don't know</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="leave">Leave</Form.Label>
          <Form.Control
            as="select"
            name="leave"
            value={inputs.leave}
            onChange={handleChange}
          >
            <option value="Very easy">Very easy</option>
            <option value="Somewhat easy">Somewhat easy</option>
            <option value="Somewhat difficult">Somewhat difficult</option>
            <option value="Very difficult">Very difficult</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="mental_health_consequence">
            Mental Health Consequence
          </Form.Label>
          <Form.Control
            as="select"
            name="mental_health_consequence"
            value={inputs.mental_health_consequence}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="Maybe">Maybe</option>
            <option value="No">No</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="phys_health_consequence">
            Physical Health Consequence
          </Form.Label>
          <Form.Control
            as="select"
            name="phys_health_consequence"
            value={inputs.phys_health_consequence}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="Maybe">Maybe</option>
            <option value="No">No</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="coworkers">Coworkers</Form.Label>
          <Form.Control
            as="select"
            name="coworkers"
            value={inputs.coworkers}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="Some of them">Some of them</option>
            <option value="No">No</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="supervisor">Supervisor</Form.Label>
          <Form.Control
            as="select"
            name="supervisor"
            value={inputs.supervisor}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="Some of them">Some of them</option>
            <option value="No">No</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="mental_health_interview">
            Mental Health Interview
          </Form.Label>
          <Form.Control
            as="select"
            name="mental_health_interview"
            value={inputs.mental_health_interview}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="Maybe">Maybe</option>
            <option value="No">No</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="phys_health_interview">
            Physical Health Interview
          </Form.Label>
          <Form.Control
            as="select"
            name="phys_health_interview"
            value={inputs.phys_health_interview}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="Maybe">Maybe</option>
            <option value="No">No</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label htmlFor="mental_vs_physical">
            Mental vs Physical Health
          </Form.Label>
          <Form.Control
            as="select"
            name="mental_vs_physical"
            value={inputs.mental_vs_physical}
            onChange={handleChange}
          >
            <option value="Yes">Yes</option>
            <option value="Maybe">Maybe</option>
            <option value="No">No</option>
          </Form.Control>
        </FormGroup>

        <FormGroup>
          <Form.Label>Observed Consequence</Form.Label>
          <Form.Check
            type="radio"
            label="Yes"
            name="obs_consequence"
            value="Yes"
            checked={inputs.obs_consequence === "Yes"}
            onChange={handleChange}
          />
          <Form.Check
            type="radio"
            label="No"
            name="obs_consequence"
            value="No"
            checked={inputs.obs_consequence === "No"}
            onChange={handleChange}
          />
        </FormGroup>

        <Button type="submit">Submit</Button>
      </Form>
    </div>
  );
}

export default Predictor;
