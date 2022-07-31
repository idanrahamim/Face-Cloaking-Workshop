function init() {
    const btn = document.querySelector("#btn");

    btn.addEventListener("click", (event) => {
        if (document.querySelector("#inputPrivacy1").value ===
            document.querySelector("#inputPrivacy2").value){

            event.preventDefault();
            document.querySelector("h2").insertAdjacentHTML('afterend',
                `<ul><li id="dpl">Select different privacy levels</li></ul>`);
        }
        else if (document.querySelector("#file").value !== ""){
            const dpl = document.querySelector("#dpl");
            if (dpl){dpl.remove();}

            document.querySelector(".container").insertAdjacentHTML('beforeend',
                `<div id="loading-box">
                <div class="loading"></div>
                <h4><br>~ LOADING ~<br> <br>Process can take up to 5 minutes</h4>
                </div>`);
        }
    });
}

init()
