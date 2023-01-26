using FibonacciAppMVC.Data;
using FibonacciAppMVC.Models;
using Microsoft.EntityFrameworkCore;
using System.Numerics;

namespace FibonacciAppMVC.Services
{
    public class SearchHistoryService : ISearchHistoryService
    {
        private readonly AppDbContext _db;

        /// <summary>
        /// The constructor method initializes the _db context variable
        /// </summary>
        /// <param name="db">AppDbContext</param>
        public SearchHistoryService(AppDbContext db)
        {
            _db = db;
        }
        /// <summary>
        /// This method returns the search history
        /// </summary>
        /// <returns></returns>
        public async Task<List<Search>> GetSearchHistoryAsync()
        {
            try
            {
                // Get the search history
                var history = await _db.SearchHistory.OrderByDescending(e => e.Date).ToListAsync();

                return history;
            }

            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }

        /// <summary>
        /// This method saves a search in the database
        /// </summary>
        /// <param name="num">The searched number</param>
        /// <exception cref="Exception"></exception>
        public async Task SaveSearchHistoryAsync(BigInteger num)
        {
            try
            {
                // Create a search object
                Search searchedNum = new Search();
                searchedNum.Num = num.ToString();
                searchedNum.Date = DateTime.Now;
                searchedNum.Id = new Guid();
                // Add the search record to the set
                await _db.SearchHistory.AddAsync(searchedNum);
                // Save changes
                _db.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }

    }
}
