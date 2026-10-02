d = 12;
function f(){
    var a = 62;
    f2();
    function f2(){
        var b = 45;
        f3();
        function f3(){
            console.log(a);
            console.log(b);
            console.log(d);

     }
  }   
}
var d;
f();