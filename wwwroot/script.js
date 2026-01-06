const form = document.getElementById('sleepForm');
const predictBtn = document.getElementById('predictBtn');
const retrainBtn = document.getElementById('retrainBtn');
const resultDiv = document.getElementById('result');
const retrainResultDiv = document.getElementById('retrainResult');
const retrainFormDiv = document.getElementById('retrainForm');
const confirmRetrainBtn = document.getElementById('confirmRetrainBtn');
const cancelRetrainBtn = document.getElementById('cancelRetrainBtn');

let currentFormData = null;

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        dailySteps: parseInt(document.getElementById('dailySteps').value),
        physicalActivity: parseInt(document.getElementById('physicalActivity').value),
        stressLevel: parseInt(document.getElementById('stressLevel').value),
        sleepDuration: parseFloat(document.getElementById('sleepDuration').value)
    };

    currentFormData = formData;

    predictBtn.disabled = true;
    resultDiv.classList.add('hidden');
    retrainResultDiv.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        
        document.getElementById('quality').textContent = data.quality;
        document.getElementById('advice').textContent = data.advice;
        resultDiv.classList.remove('hidden');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        predictBtn.disabled = false;
    }
});

retrainBtn.addEventListener('click', () => {
    if (!currentFormData) {
        alert('Please submit the form first to get a prediction.');
        return;
    }

    retrainFormDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    retrainResultDiv.classList.add('hidden');
});

cancelRetrainBtn.addEventListener('click', () => {
    retrainFormDiv.classList.add('hidden');
    document.getElementById('qualityOfSleep').value = '';
});

confirmRetrainBtn.addEventListener('click', async () => {
    if (!currentFormData) {
        return;
    }

    const qualityOfSleep = parseInt(document.getElementById('qualityOfSleep').value);
    
    if (qualityOfSleep < 1 || qualityOfSleep > 10) {
        alert('Quality of Sleep must be between 1 and 10');
        return;
    }

    const retrainData = {
        ...currentFormData,
        qualityOfSleep: qualityOfSleep
    };

    confirmRetrainBtn.disabled = true;

    try {
        const response = await fetch('/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(retrainData)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: 'Retrain failed' }));
            throw new Error(errorData.message || 'Retrain failed');
        }

        const data = await response.json();
        
        document.getElementById('retrainMessage').textContent = data.message;
        retrainResultDiv.classList.remove('hidden');
        retrainFormDiv.classList.add('hidden');
        document.getElementById('qualityOfSleep').value = '';
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        confirmRetrainBtn.disabled = false;
    }
});

