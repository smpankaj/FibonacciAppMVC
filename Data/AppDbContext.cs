using Microsoft.EntityFrameworkCore;
using FibonacciAppMVC.Models;

namespace FibonacciAppMVC.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }

        public DbSet<Search> SearchHistory { get; set; }

    }
}



