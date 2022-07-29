function init() {
    const btn = document.querySelector("#btn");

    btn.addEventListener("click", () => {

        if (document.querySelector("#file").value !== ""){
            document.querySelector(".container").insertAdjacentHTML('beforeend',
                `<div id="loading-box">
                <div class="loading"></div>
                <h4><br>~ LOADING ~<br> <br>Process can take up to 5 minutes</h4>
                </div>`);
        }
    });
}

init()

