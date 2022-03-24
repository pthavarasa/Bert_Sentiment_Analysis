const getInputContent = () => inputBtn.value;
const existClass = (el, classEl) => [...el.classList].includes(classEl)
const setActiveBtn = (el, otherEl) => {
    error.innerHTML = ""
    if (emotion !== el.value) {
        emotion = el.value;
        if (!existClass(el.target, el.class)) el.target.classList.toggle(el.class)
        if (existClass(otherEl.target, otherEl.class)) otherEl.target.classList.toggle(otherEl.class)
    }
}
const setupLog = (el, otherEl, value) => {
    otherEl.innerHTML = "";
    el.innerHTML = value;
}

const submitBtn = document.getElementById('submit');
const goodBtn = document.getElementById('good');
const badBtn = document.getElementById('bad');
const inputBtn = document.getElementById('input');
const error = document.getElementById('error');
const succes = document.getElementById('succes');
const good = { target: goodBtn, class: 'good-selected', value: 1 }
const bad = { target: badBtn, class: 'bad-selected', value: 0 }
let emotion = -1;

goodBtn.addEventListener('click', () => {setActiveBtn(good, bad)})
badBtn.addEventListener('click', () => {setActiveBtn(bad, good)})

submitBtn.addEventListener('click', () => {
    setupLog(error, succes, "")
    if (emotion > -1 && getInputContent()) {
        //fetch(`http://localhost:8000/text/${getInputContent()}?emotion=${emotion}`).then(response => {
        fetch(`http://localhost:8000/text/${getInputContent()}/${emotion}`).then(response => {
            response.json().then(parsedJson => {
                setupLog(succes, error, "Predicted : " + parsedJson.sentiment)
            })
        }).catch(err => {
            setupLog(error, succes, "Error during communication with server")
        })
        emotion = -1;
        if (existClass(good.target, good.class)) good.target.classList.remove(good.class)
        else if (existClass(bad.target, bad.class)) bad.target.classList.remove(bad.class)
    } else {
        if (emotion < 0) setupLog(error, succes, "You must specify emotion")
        else if (!getInputContent()) setupLog(error, succes, "You must specify sentence")
    }
})