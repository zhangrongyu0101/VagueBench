const turns = [
  ['“Are you looking to invest, collect, or just research?”', '“Invest, to hedge inflation.”'],
  ['“What is your budget, risk level, and time horizon?”', '“$500K, medium risk, and 2–3 years.”'],
  ['“What do you hold now? Would you prefer ETFs or physical metal?”', '“Stocks and funds. I prefer ETFs.”']
];
document.querySelectorAll('[data-turn]').forEach(button => {
  button.addEventListener('click', () => {
    const [question, answer] = turns[Number(button.dataset.turn)];
    document.querySelector('#dialogue .question').textContent = question;
    document.querySelector('#dialogue .answer').textContent = answer;
    document.querySelectorAll('[data-turn]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  });
});
