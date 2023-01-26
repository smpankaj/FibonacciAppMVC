$(document).ready(function () {
    // Get the search history
    getHistory();
});

function getFib() {
    // Clear the table that displays search history
    $("#lblFib").html("");
    var num = $('#txtNum').val();
    //If the input is invalid, alert the user
    if (!validateInput(num)) {
        alert("Please enter a valid number");
        return;
    } 
    num = parseInt(num);
    if (num < 1) {
        alert("The number should be greater than 0");
        return;
    }
    var url = "/Home/GetNthFib?num=" + num;
    $.ajax({
        url: url,
        type: 'POST',
        contentType: 'application/json',
        success: function (response) {
            console.log("success")
            $("#lblFib").html("Fibonacci number is " + response);
            getHistory();
        },
        error: function (response) {
            console.log(response);
            console.log(JSON.parse(response));
        }
    });  
}

// The following function gets the search history
function getHistory() {
    var url = "/Home/GetHistory";
    $.ajax({
        url: url,
        type: 'GET',
        contentType: 'application/json',
        success: function (response) {
            // Populate the table with the history
            populateHistory(response);
        },
        error: function (response) {
            console.log(response);
            console.log(JSON.parse(response));
        }
    });
}

// The following function populates the table on the home page with search history
function populateHistory(response) {
    if (!response) {
        $('#tbl').html("No history of searches found");
        return;
    }
    data = JSON.parse(response);
    var htmlStr = "<tr><th scope='col'>Number</th> <th scope='col'>Date</th></tr>";
    for (var i = 0; i < data.length; i++) {
        htmlStr += "<tr><td>" + data[i].Num + "</td><td>" + data[i].Date + "</td></tr>";
    }
    $('#tbl').html(htmlStr);
}

// The following function validat whether the entered param is num
function validateInput(num) {
    if (num && $.isNumeric(num) && Math.floor(num) == num)
        return true;
    return false;
}
