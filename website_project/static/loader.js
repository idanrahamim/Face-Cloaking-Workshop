
function init() {
    let submitted_flag = false;
    const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif'];
    const btn = document.querySelector("#btn");
    document.querySelector("#title2").insertAdjacentHTML('afterend', `<ul id="error_list"></ul>`);

    btn.addEventListener("click", (event) => {

        if (submitted_flag === true){
            event.preventDefault();
        }
        else{
            let clear_flag = true;

            const file = document.querySelector("#file");
            const extension = file.value.split(".").pop().toLowerCase();
            const allow = document.querySelector("#allow");
            if(!(ALLOWED_EXTENSIONS.includes(extension))){
                clear_flag = false;
                event.preventDefault();
                file.value='';
                if (!allow){
                    document.querySelector("#error_list").insertAdjacentHTML('beforeend',
                    `<li id="allow">Allowed image types are - png, jpg, jpeg, gif</li>`);
                }
            }else if(allow) {allow.remove();}

            const equal = document.querySelector("#equal");
            if(document.querySelector("#inputPrivacy1").value ===
            document.querySelector("#inputPrivacy2").value){
                clear_flag = false;
                event.preventDefault();
                if (!equal){
                    document.querySelector("#error_list").insertAdjacentHTML('beforeend',
                    `<li id="equal">Select different privacy levels</li>`);
                }
            }else if(equal) {equal.remove();}

            if (clear_flag){
                submitted_flag = true;
                [... document.querySelectorAll('ul')].map(i => i.innerHTML = '');
                document.querySelector(".container").insertAdjacentHTML('beforeend',
                    `<div id="loading-box">
                    <div class="loading"></div>
                    <h4><br>~ LOADING ~<br> <br>Process can take up to 5 minutes</h4>
                    </div>`);
            }
        }
    })
}

init()
