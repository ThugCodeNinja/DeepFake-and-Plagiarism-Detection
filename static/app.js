document.addEventListener('DOMContentLoaded', function () {
    const submitButton = document.getElementById('submitButton');
    const textArea = document.getElementById('text');
    submitButton.addEventListener('click', function () {
        const text = textArea.value;

        // Make an HTTP request to your Node.js environment
        fetch('http://localhost:5000/chatgptdetection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            mode: 'cors',
            body: JSON.stringify({ text }),
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            // Process the response from your Node.js environment
        })
        .catch(error => {
            console.error(error);
        });
    });
});













// import {HfInference} from "@huggingface/inference";
// import dotenv from "dotenv"
// import axios from "axios";

// dotenv.config()


// const HF_ACCESS_TOKEN='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'
// const inference= new HfInference(HF_ACCESS_TOKEN)

// const model='thugCodeNinja/robertafinetune';
// const result= await inference.textClassification({
//     inputs: input,
// });

// console.log(result);

