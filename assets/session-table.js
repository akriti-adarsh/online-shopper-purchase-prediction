export default function({parentElement, data, setTriggerValue}) {
  const host = parentElement.querySelector('.sessions-host');
  host.innerHTML = data.html;
  const onClick = event => {
    const button = event.target.closest('button[data-session]');
    if (button) setTriggerValue('selected', button.dataset.session);
    const sort = event.target.closest('button[data-sort]');
    if (sort) setTriggerValue('sort', sort.dataset.sort);
  };
  host.addEventListener('click', onClick);
  return () => host.removeEventListener('click', onClick);
}
