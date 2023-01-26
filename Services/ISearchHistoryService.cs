using FibonacciAppMVC.Models;
using Microsoft.EntityFrameworkCore;
using System.Numerics;

namespace FibonacciAppMVC.Services
{
    public interface ISearchHistoryService
    {
        Task<List<Search>> GetSearchHistoryAsync();
        Task SaveSearchHistoryAsync(BigInteger num);


    }
}
