const express = require("express");
const app_project = express.Router();

app_project.post("/predict", async (req, res) => {
    const {
        Age,
        Gender,
        Country,
        state,
        self_employed,
        family_history,
        work_interfere,
        no_employees,
        remote_work,
        tech_company,
        benefits,
        care_options,
        wellness_program,
        seek_help,
        anonymity,
        leave,
        mental_health_consequence,
        phys_health_consequence,
        coworkers,
        supervisor,
        mental_health_interview,
        phys_health_interview,
        mental_vs_physical,
        obs_consequence,
        comments,
    } = req.body;
    import("@gradio/client").then(async (module) => {
        const client = module.client;
        // rest of your code
        const app = await client(
            "https://rajababu15-ht-bk-gr.hf.space/--replicas/kk9xd/"
        );
        const result = await app.predict("/predict", [
            new Date().toISOString().slice(0, 19).replace("T", " "),
            Age,
            Gender,
            Country,
            state,
            self_employed,
            family_history,
            work_interfere,
            no_employees,
            remote_work,
            tech_company,
            benefits,
            care_options,
            wellness_program,
            seek_help,
            anonymity,
            leave,
            mental_health_consequence,
            phys_health_consequence,
            coworkers,
            supervisor,
            mental_health_interview,
            phys_health_interview,
            mental_vs_physical,
            obs_consequence,
            comments,
        ]);
        res.status(200).json({data:result.data});
    });
});

module.exports = app_project;
