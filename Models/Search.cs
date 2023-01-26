using System.ComponentModel.DataAnnotations;
using System.Numerics;

namespace FibonacciAppMVC.Models
{
    public class Search
    {
        [Key]
        public Guid Id { get; set; }
        public string Num { get; set; }
        public DateTime Date { get; set; }
    }
}
