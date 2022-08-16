
function init_index() {
    /*
    init onload function for index.html home page.
    this function listens to the submit button, validates the file,the privacy buttons selections and displays errors.
    it displays the loading bar on button click.
     */
    let submitted_flag = false; // false if the form has already submitted, otherwise true
    const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif'];
    const btn = document.querySelector("#btn");
    document.querySelector("#title2").insertAdjacentHTML('afterend', `<ul id="error_list"></ul>`); //creates error list

    btn.addEventListener("click", (event) => { // listens to submit button

        if (submitted_flag === true){
            event.preventDefault();
        }
        else{
            let clear_flag = true; // true if all inputs found valid, false if errors.

            const file = document.querySelector("#file");
            const extension = file.value.split(".").pop().toLowerCase();
            const allow = document.querySelector("#allow");
            if(!(ALLOWED_EXTENSIONS.includes(extension))){ // image type check
                clear_flag = false;
                event.preventDefault();
                file.value='';
                if (!allow){ // displays error if not displayed yet
                    document.querySelector("#error_list").insertAdjacentHTML('beforeend',
                    `<li id="allow">Allowed image types are - png, jpg, jpeg, gif</li>`);
                }
            }else if(allow) {allow.remove();} // removes error if no invalid extension yet

            const equal = document.querySelector("#equal"); // enforces unequal privacy levels selections
            if(document.querySelector("#inputPrivacy1").value ===
            document.querySelector("#inputPrivacy2").value){
                clear_flag = false;
                event.preventDefault();
                if (!equal){ // displays error if not displayed yet
                    document.querySelector("#error_list").insertAdjacentHTML('beforeend',
                    `<li id="equal">Select different privacy levels</li>`);
                }
            }else if(equal) {equal.remove();} // removes error if no equal selection yet

            if (clear_flag){ // if no errors send the form and display the loading bar.
                submitted_flag = true;
                [... document.querySelectorAll('ul')].map(i => i.innerHTML = '');
                document.querySelector(".container").insertAdjacentHTML('beforeend',
                    `<div id="loading-box">
                    <div class="loading"></div>
                    <h4><br>~ LOADING ~<br> <br>Process can take up to a minute</h4>
                    </div>`);
            }
        }
    })
}

function init_submission(){
    /*
    init onload function for submission.html results page.
    this function listens to download buttons clicks, executes the user-side download image,
    and enforces to send the download clicks information form at most once for each displayed privacy level.
     */
    const btns_flag = [false, false, false];
    const form = document.querySelector('#form');
    document.querySelectorAll('input').forEach(btn => { // listens to all buttons
     btn.addEventListener("click", (event) => { // handle button click
        event.preventDefault(); // prevents default submit button operation of sending the form
        const img = document.querySelector("#img"+btn.id).src;
        document.body.insertAdjacentHTML('beforeend', `<a id="lin" href="${img}" download="${btn.name}"></a>`); //create download link
        const lin = document.querySelector('#lin');
        lin.click(); // executes download
        lin.remove(); // removes download link
        if(btns_flag[btn.id-1] === false){
            btns_flag[btn.id-1] = true;
            form.requestSubmit(btn); // sends the form by simulating button click
        }
     });
    });
}

if(document.title === 'Face Cloaking'){ // index.html
    init_index();}
else{ // submissions.html
    init_submission();}
