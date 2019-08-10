// getElementById
function $id(id) {
    return document.getElementById(id);
}

//
// output information
function Output(msg) {
    var m = $id("messages");
    m.innerHTML = msg + m.innerHTML;
}


//
// initialize
function Init() {

    let fileselect = document.getElementById("fileselect");
    console.log("fileselect: %o", fileselect);
    let filedrag = document.getElementById("filedrag");

    // file select
    fileselect.addEventListener("change", FileSelectHandler, false);


    // file drop
    filedrag.addEventListener("dragover", FileDragHover, false);
    filedrag.addEventListener("dragleave", FileDragHover, false);
    filedrag.addEventListener("drop", FileSelectHandler, false);
    filedrag.style.display = "block";
}

// file drag hover
function FileDragHover(e) {
    e.stopPropagation();
    e.preventDefault();
    e.target.className = (e.type == "dragover" ? "hover" : "");
}

// file selection
function FileSelectHandler(e) {

    console.log("drop: %o", e);

    // cancel event and hover styling
    FileDragHover(e);

    if (e.dataTransfer && e.dataTransfer.items) {
        for (let item of e.dataTransfer.items) {
            console.log("item: %o", item);
            if (item.kind == 'file') {
                ParseFile(item.getAsFile());
            } else if (item.type == 'text/uri-list') {
                item.getAsString(text => {
                    console.log("item text: %s", text);
                    ParseUrl(text);
                });
            }
        }
    } else if (e.target.files) {
        for (let file of e.target.files) {
            ParseFile(file);
        }
    }
}

function ParseUrl(url) {
    let image = document.getElementById("image");
    image.innerHTML =
        `<p><strong>${url}</strong><br />` +
        `<img src="${url}"/></p>`;

    let div = document.getElementById("caption");
    div.innerText = "Erzeuge Bildbeschreibung ...";

    let formData = new FormData();
    formData.append("url", url);
    fetchCaption(formData);

}

function ParseFile(file) {

    console.log("file: %o", file);

    // display an image
    if (file.type.indexOf("image") == 0) {
        var reader = new FileReader();
        reader.onload = (e) => {
            let image = document.getElementById("image");
            image.innerHTML =
                "<p><strong>" + file.name + ":</strong><br />" +
                '<img src="' + e.target.result + '" /></p>';
        }
        reader.readAsDataURL(file);
    }

    let div = document.getElementById("caption");
    div.innerText = "Erzeuge Bildbeschreibung ...";

    let formData = new FormData();
    formData.append("image", file);
    fetchCaption(formData)
}

function fetchCaption(body) {
    let request = new Request("./caption");

    fetch(request, {
            method: "POST",
            body: body
        })
        .then(response => {
            if (response.status == 200 || response.status == 201) {
                response.json()
                    .then(json => {
                        console.log(json);
                        let div = document.getElementById("caption");
                        div.innerText = json.annotation;
                    });
            } else {
                let div = document.getElementById("caption");
                div.innerText = `Oops, da ist etwas schiefgegangen, Status=${response.status}`;
            }
        });
}

window.onload = Init;