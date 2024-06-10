const express = require("express");
const bodyParser = require("body-parser");
const mongoose = require("mongoose");

// MongoDB-Verbindung herstellen
mongoose.connect("mongodb+srv://chahidhamdaoui:hamdaoui1@cluster0.kezbfis.mongodb.net/test?retryWrites=true&w=majority", {
    useNewUrlParser: true,
    useUnifiedTopology: true
});
const db = mongoose.connection;
db.on("error", console.error.bind(console, "MongoDB-Verbindungsfehler:"));
db.once("open", function() {
    console.log("Verbunden mit MongoDB-Datenbank.");
});

// Mongoose-Modell für Bewerbungen definieren
const Application = mongoose.model("Application", {
    firstName: String,
    lastName: String,
    age: Number,
    skills: String,
    job: String,
    hobbies: String
});

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));

// POST-Anfrage zum Speichern der Bewerbung
app.post("/submitApplication", function(req, res) {
    const { firstName, lastName, age, skills, job, hobbies } = req.body;
    const application = new Application({ firstName, lastName, age, skills, job, hobbies });
    application.save(function(err) {
        if (err) {
            console.error("Fehler beim Speichern der Bewerbung:", err);
            res.status(500).send("Fehler beim Speichern der Bewerbung.");
        } else {
            console.log("Bewerbung erfolgreich gespeichert:", application);
            res.sendStatus(200);
        }
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, function() {
    console.log(`Server läuft auf http://localhost:${PORT}`);
});
